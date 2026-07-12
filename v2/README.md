# Mining Bot v2

Refatoração do bot de mineração — **sem mapa virtual no v0**.

## Documentação

Leia **[ARCHITECTURE.md](./ARCHITECTURE.md)** antes de implementar qualquer módulo.

## Status

✅ **Fase 1** — loop funcional: percepção + brain + walker + preview + gravação.

## Comandos

```powershell
cd mining_bot
python -m v2.main --preview
python -m v2.main          # F6 liga
python -m v2.calibrate_gray  # cor do nó cinza → v2/config.json
```

Controles: **F6** liga/desliga | **F7** pausa | **F9** sai | **E/F8** próximo | **Q** sai preview

### Calibrar nó cinza (evitar confundir com estrada)

1. `python -m v2.main --preview` (ou `python -m v2.calibrate_gray`)
2. No preview: tecla **G**, depois clique no **disco branco** do nó no minimapa
3. Cor salva em `tier_colors_hsv.gray` dentro de `v2/config.json`
4. O detector usa só esses bounds HSV (`gray_achromatic_expand: false`)

## v1 legado

Código antigo permanece na pasta pai (`simple_bot.py`, `tile_map.py`, etc.) até v0 passar nos critérios de aceite.
