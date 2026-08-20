import { AlertCircle, CheckCircle2, X } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { useI18n } from '../i18n';

export type NotificationNotice = {
  tone: 'success' | 'error';
  title: string;
  message: string;
};

type Props = NotificationNotice & {
  onDismiss: () => void;
  compact?: boolean;
};

export function NotificationToast({ tone, title, message, onDismiss, compact = false }: Props) {
  const { t } = useI18n();
  const dismissRef = useRef(onDismiss);
  dismissRef.current = onDismiss;

  useEffect(() => {
    const timeout = window.setTimeout(() => dismissRef.current(), tone === 'success' ? 4500 : 7000);
    return () => window.clearTimeout(timeout);
  }, [message, tone]);

  return (
    <div
      className={`notification-toast ${tone}${compact ? ' compact' : ''}`}
      role={tone === 'error' ? 'alert' : 'status'}
      aria-live="polite"
      aria-label={`${title}: ${message}`}
    >
      {tone === 'success' ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
      <div>
        <strong>{title}</strong>
        <span title={message}>{message}</span>
      </div>
      <button type="button" aria-label={t('Dismiss notification')} onClick={onDismiss}>
        <X size={15} />
      </button>
    </div>
  );
}
