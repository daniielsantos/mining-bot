# Mining Bot v2 — Arquitetura

## Por que refatorar

A v1 acumulou camadas que tentam resolver o mesmo problema:

```
blip na tela → ID persistente → nó virtual → tile estático → phaseCorrelate
→ landmark → snap → anti-snap → bearing tile vs bearing tela
```

Cada camada adicionou um sistema de coordenadas. Quando se misturam, aparecem:
linha congelada, distância `T` que não cai, rumo ~90°, andar de lado.

**Lição:** mapa virtual só faz sentido *depois* que navegação por blip visível funciona de forma confiável.

---

## Ideia em uma frase

Bot que lê o **minimapa** (blips cinza + seta), **caminha** até o alvo com W/A/D, **interage** com E, e **detecta** a barra de mineração no HUD — tudo via captura de tela.

---

## Princípios (regras que evitam os bugs antigos)

| # | Regra | Motivo |
|---|--------|--------|
| 1 | **Navegação só em coordenadas de tela** (pivot = jogador) | Minimapa rola/gira; tile “imaginário” diverge fácil |
| 2 | **Um alvo travado por vez** — troca só com E/F8 ou após minerar | Anti-switch na aproximação |
| 3 | **Duas fases de movimento:** `ALIGN` (só A/D) → `WALK` (W) | Evita andar de lado (W+D com rumo grande) |
| 4 | **Percepção → Decisão → Ação** — módulos separados, dados imutáveis por frame | Sem `_tracked_x` espalhado |
| 5 | **Debug desde o dia 1** — overlay + JSON + replay offline | Diagnóstico sem adivinhar |
| 6 | **Mapa virtual é v1 opcional**, não parte do core | Só entra quando v0 passar nos critérios |

---

## O que NÃO entra no v0

- `tile_map.py`, `session_scanner.py`, phaseCorrelate, landmark
- Projeção de nós fixos no tile
- `bearing_error_deg` / rumo em espaço tile para teclas
- Ingest incremental de centenas de nós no mapa

---

## Pipeline por frame

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌────────────┐
│   Capture   │───▶│    Perception    │───▶│    Brain    │───▶│  Control   │
│ minimap+HUD │    │ arrow, blips, ui │    │ state machine│    │ W/A/D/E    │
└─────────────┘    └──────────────────┘    └─────────────┘    └────────────┘
                            │                      │
                            └──────────┬───────────┘
                                       ▼
                              ┌─────────────────┐
                              │  Debug overlay  │
                              │  + recorder     │
                              └─────────────────┘
```

### `FrameContext` (único objeto por tick)

Todo módulo recebe/devolve dados via este contrato — **sem globals mutáveis**.

```python
@dataclass(frozen=True)
class FrameContext:
    tick: int
    timestamp: float
    minimap_bgr: np.ndarray
    hud_bgr: np.ndarray
    pivot: tuple[float, float]          # centro da seta (px)
    arrow: ArrowState                   # tip, facing, detected
    blips: tuple[Blip, ...]             # só tiers permitidos (gray)
    hud: HudState                       # mining_active, progress_pct
    lock: TargetLock | None             # alvo travado
    phase: Phase                        # SCAN | GOTO | INTERACT | MINING | COOLDOWN
    bearing_deg: float | None           # seta → alvo (tela)
    dist_px: float                      # pivot → alvo (tela)
    action: str                         # idle | align-a | forward | interact-e
    meta: dict                          # debug extras
```

---

## Módulos

### `capture/`
- Responsabilidade: MSS, ROIs do config, FPS limit
- Não interpreta imagem

### `perception/`
- **`arrow`**: detecta seta, pivot, ponta, `facing_deg` (reusa `minimap_tracker` da v1)
- **`blips`**: detecta círculos cinza, filtra exclusão do jogador (reusa lógica de `node_detector.scan_blips`)
- **`hud`**: barra de progresso / label de mineração (reusa `screen_ui`)

Saída: structs puras, sem estado de navegação.

### `navigation/`
- **`bearing`**: `heading_error = walk_heading_from_arrow(arrow, target_x, target_y)` — **única** fonte de rumo para teclas
- **`target_lock`**: escolhe/trava/rastreia blip por proximidade + `track_id` simples (posição suavizada na tela)
- **`walker`**: máquina ALIGN | WALK — **nunca W se |bearing| > turn_walk_deg**

Regras do lock:
1. `lock(blip)` — trava o blip mais próximo acima de `min_pick_px`
2. Enquanto travado, `track()` atualiza posição do blip na tela (blob matching)
3. Se perder por N frames → `unlock()` → volta SCAN
4. E/F8 → `lock_next()` (próximo blip mais próximo, excluindo done)

**Sem coordenadas tile.** Distância de chegada = `dist_px` na tela (`arrive_px`).

### `brain/`
- Máquina de estados finita, ~80 linhas
- Transições:

```
SCAN ──(lock ok)──▶ GOTO
GOTO ──(dist≤arrive ∧ |brg|≤align)──▶ INTERACT
INTERACT ──(hud mining)──▶ MINING
MINING ──(hud idle)──▶ COOLDOWN
COOLDOWN ──(timeout)──▶ SCAN ou lock_next → GOTO
```

- Não chama OpenCV; só lê `FrameContext` e decide `phase` + pedidos ao walker

### `debug/`
- **`overlay`**: minimap + linha verde pivot→alvo + barra de status (3 linhas max)
- **`recorder`**: wrap de `SessionFrameRecorder` + schema JSON fixo
- **`replay`**: (fase 2) reprocessa JSON+jpg sem GTA

---

## Reuso da v1 (copiar/adaptar, não importar acoplado)

| v1 | v2 uso |
|----|--------|
| `keyboard_input.py` | direto |
| `config.py` + ROIs | adaptar `v2/config.json` |
| `minimap_tracker.py` | via `perception/arrow.py` |
| `node_detector.scan_blips` | via `perception/blips.py` |
| `screen_ui` | via `perception/hud.py` |
| `navigator.walk_heading_from_arrow` | via `navigation/bearing.py` |
| `debug_capture.SessionFrameRecorder` | via `debug/recorder.py` |
| `display.add_status_bar` | via `debug/overlay.py` |

**Não reutilizar:** `tile_map`, `session_scanner`, `simple_bot`, `bot.py` (monolitos)

---

## Critérios de aceite v0

Antes de qualquer mapa virtual:

1. **30s de GOTO** em área com 2–3 nós cinza: `dist_px` cai monotonicamente (±ruído 3px)
2. **|bearing| < 20°** antes de `action=forward` (sem W+A/D simultâneo prolongado)
3. **Lock estável** — não troca alvo na aproximação
4. **Interação** — E dispara, HUD detecta mineração em ≤2s
5. **Replay** — sessão gravada reprocessa sem crash

---

## Roadmap

### Fase 0 — Esqueleto (atual)
- [x] ARCHITECTURE.md
- [ ] `core/types.py` contratos
- [ ] stubs com docstrings
- [ ] `config.example.json`

### Fase 1 — v0 Blip Bot
- [ ] capture + perception funcionando
- [ ] target_lock + bearing + walker
- [ ] brain loop + main.py
- [ ] overlay + recorder

### Fase 2 — Polimento
- [ ] replay offline
- [ ] testes com fixtures
- [ ] calibrate pivot CLI

### Fase 3 — v1 Mapa (opcional, só se v0 OK)
- [ ] grafo de nós visitados (posição relativa ao bootstrap, não correlacionar areia)
- [ ] memória “já minerado” entre sessões
- [ ] **nunca** substituir rumo de tela por rumo tile para W/A/D

---

## Estrutura de pastas

```
mining_bot/v2/
  ARCHITECTURE.md          ← este documento
  README.md
  main.py                  ← python -m v2.main [--preview]
  config.example.json

  core/
    types.py               ← FrameContext, Blip, ArrowState, TargetLock, Phase
    config.py

  capture/
    grabber.py

  perception/
    arrow.py
    blips.py
    hud.py
    pipeline.py            ← monta FrameContext parcial

  navigation/
    bearing.py
    target_lock.py
    walker.py

  brain/
    states.py
    tick.py                ← um tick: perceive → decide → act

  debug/
    overlay.py
    recorder.py
    replay.py              ← stub fase 2

  tests/
    test_bearing.py
    test_target_lock.py
    fixtures/
```

---

## Schema JSON de debug (fixo)

```json
{
  "tick": 42,
  "phase": "GOTO",
  "action": "align-d",
  "bearing_deg": -34.2,
  "dist_px": 48.1,
  "lock_id": 1,
  "lock_x": 180.0,
  "lock_y": 95.0,
  "blips": 2,
  "arrow_ok": true,
  "game_focus": true
}
```

Sem campos `player_tile_x`, `pure_turn`, `landmark` — v0 não precisa.
