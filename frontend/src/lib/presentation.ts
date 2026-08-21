const INTERNAL_RESOURCE_PATTERN = /(?:view:)?(?:insight|analysis|evidence|derived_evidence|forecast|anomaly|visualization):[A-Za-z0-9_.:#=-]+|\b(?:ins_ana|insight|ana|evi|dev_ana|viz)_[A-Za-z0-9_.:-]+/gi;

const INTERNAL_METADATA_KEYS = new Set([
  'analysis_id',
  'analysis_ids',
  'artifact_id',
  'artifact_ids',
  'binding_id',
  'binding_ids',
  'derived_evidence_id',
  'derived_evidence_ids',
  'evidence_id',
  'evidence_ids',
  'input_evidence_id',
  'input_source_refs',
  'insight_id',
  'insight_ids',
  'item_id',
  'item_ids',
  'source_id',
  'source_ids',
  'source_ref',
  'source_refs',
]);

export function containsInternalIdentifier(value: string | null | undefined): boolean {
  if (!value) return false;
  return new RegExp(INTERNAL_RESOURCE_PATTERN.source, 'i').test(value.trim());
}

export function sanitizeUserFacingText(value: string): string {
  const replacement = /[\u3400-\u9fff]/.test(value) ? '分析结果' : 'analysis result';
  return value
    .replace(new RegExp(INTERNAL_RESOURCE_PATTERN.source, 'gi'), replacement)
    .replace(/(?:分析结果)(?:[\s、,，/]+分析结果)+/g, '分析结果')
    .replace(/(?:analysis result)(?:[\s,，/]+analysis result)+/gi, 'analysis result');
}

export function sanitizeEvidenceForDisplay(value: unknown): unknown {
  if (Array.isArray(value)) {
    const items = value
      .map((item) => sanitizeEvidenceForDisplay(item))
      .filter((item) => item !== undefined);
    return items.length > 0 ? items : undefined;
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).flatMap(([key, item]) => {
      if (INTERNAL_METADATA_KEYS.has(key.toLowerCase())) return [];
      const sanitized = sanitizeEvidenceForDisplay(item);
      return sanitized === undefined ? [] : [[key, sanitized] as const];
    });
    return entries.length > 0 ? Object.fromEntries(entries) : undefined;
  }
  if (typeof value === 'string') {
    if (containsInternalIdentifier(value) && sanitizeUserFacingText(value).trim() === (/\p{Script=Han}/u.test(value) ? '分析结果' : 'analysis result')) {
      return undefined;
    }
    return sanitizeUserFacingText(value);
  }
  return value;
}
