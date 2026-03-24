# Frontend - Drug Interaction Explorer

React + TypeScript frontend for visualizing drug-drug interactions.

## Tech Stack

- **React 19** with **TypeScript**
- **Vite** - Fast build tool
- **Tailwind CSS v4** - Utility-first CSS
- **React Router** - Client-side routing
- **TanStack Query** - Server state management
- **Zustand** - Client state management
- **Cytoscape.js** - Graph visualization
- **Axios** - HTTP client

## Quick Start

### Prerequisites
- Node.js 18+ and npm

### Development

1. **Install dependencies:**
```bash
npm install
```

2. **Start dev server:**
```bash
npm run dev
```

Frontend runs at `http://localhost:3000`

3. **Build for production:**
```bash
npm run build
```

4. **Preview production build:**
```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── main.tsx           # entry point
│   ├── App.tsx            # root component
│   ├── index.css          # tailwind imports
│   └── vite-env.d.ts      # vite types
├── public/                # static assets
├── index.html             # html template
├── vite.config.ts         # vite config with proxy
├── tsconfig.json          # typescript config
└── package.json           # dependencies
```

## API Integration

The Vite dev server proxies `/api/*` requests to the backend at `http://localhost:8000`.

**Example:**
```typescript
// frontend makes request to /api/query
// vite proxies to http://localhost:8000/api/query
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Next Steps

- [ ] Set up React Router pages
- [ ] Create API service layer with TanStack Query
- [ ] Build Cytoscape.js graph component
- [ ] Implement query interface
- [ ] Add state management with Zustand
