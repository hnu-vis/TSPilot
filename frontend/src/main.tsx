import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { AppErrorBoundary } from './components/AppErrorBoundary';
import { VisualizationAuditPage } from './components/VisualizationAuditPage';
import { I18nProvider } from './i18n';
import './styles.css';

const content = window.location.pathname === '/visualization-audit'
  ? <VisualizationAuditPage />
  : <AppErrorBoundary><I18nProvider><App /></I18nProvider></AppErrorBoundary>;

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    {content}
  </React.StrictMode>,
);
