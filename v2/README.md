# Mining Bot v2

Refatoração do bot de mineração — **standalone** (não depende mais de arquivos em `mining_bot/` fora desta pasta).

## Documentação

Leia **[ARCHITECTURE.md](./ARCHITECTURE.md)** antes de implementar qualquer módulo.

## Instalação

```powershell
cd mining_bot/v2          # ou copie só a pasta v2/
pip install -r requirements.txt
```

## Comandos

```powershell
# Da pasta pai (mining_bot/) — continua funcionando:
python -m v2.main --preview

# Standalone (dentro de v2/):
python main.py --preview
python -m v2.main --preview   # se o cwd for a pasta pai do pacote v2

python -m v2.calibrate_gray
python -m v2.calibrate_camera
```

Controles: **F6** liga/desliga | **F7** pausa | **F9** sai | **E/F8** próximo | **Q** sai preview

### Calibrar nó cinza

1. `python main.py --preview` (ou `python -m v2.calibrate_gray`)
2. Tecla **G**, clique no **disco branco** do nó no minimapa
3. Cor salva em `config.json` (`tier_colors_hsv.gray`)

### Config

- Base: `config.example.json`
- Overrides locais: `config.json` (merge sobre a base)
- Não usa mais `mining_bot/config.json` da pasta pai

## Estrutura

- `vendor/` — módulos antes importados de `mining_bot/` (input, logger, detector, etc.)
- `brain/`, `navigation/`, `perception/` — lógica v2
