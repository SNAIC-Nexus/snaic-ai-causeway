# SG Taxi Weather Mesh — Web Dashboard

Real-time taxi dispatch + NEA precipitation nowcast overlay for Singapore.

## Stack

- **React 18** + **TypeScript** + **Vite 5**
- **Tailwind CSS 3** for utility styling
- **React-Leaflet 4** with CartoDB Dark Matter tiles
- All data mocked in `src/data/` — swap in real NEA / fleet API there

## Run

```bash
cd web-dashboard
npm install
npm run dev          # → http://localhost:5173
npm run build        # → dist/
```

## Structure

```
src/
├── types/index.ts          # Taxi, WeatherCell, Hotspot, NowcastStep
├── data/
│   ├── taxis.ts            # 30 mock taxis + SURGE_LOCATION
│   ├── weather.ts          # 4×5 intensity grid + INTENSITY_STYLE
│   └── hotspots.ts         # HOTSPOTS + NOWCAST_STEPS
└── components/
    ├── MapLayer.tsx         # MapContainer + dark CartoDB tiles
    ├── WeatherMesh.tsx      # Rectangle grid + area labels
    ├── TaxiMarkers.tsx      # Colored car markers + surge pill
    ├── HeaderOverlay.tsx    # Live clock + alert banner
    ├── Legend.tsx           # Gradient bar + intensity labels
    ├── StatusKey.tsx        # Taxi status key (green/yellow/blue)
    ├── NowcastTimeline.tsx  # 3-step timeline with gradient arrows
    ├── DemandHotspots.tsx   # Hotspot list with level badges
    ├── StatsPanel.tsx       # Active taxis + avg wait time
    └── BottomNav.tsx        # Tab bar + external link buttons
```

## Real API integration

Replace mock exports in `src/data/*.ts` with fetch calls to:
- **Taxis**: `https://api.data.gov.sg/v1/transport/taxi-availability`
- **Weather**: `https://api.data.gov.sg/v1/environment/2-hour-weather-forecast`
- **NEA Nowcast**: `https://api.data.gov.sg/v1/environment/rainfall`
