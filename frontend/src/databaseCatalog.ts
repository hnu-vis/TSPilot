import rawCatalog from '../../configs/database_catalog.json';

export type DatabaseExtraField = {
  key: string;
  label: string;
  input: 'text' | 'password' | 'checkbox' | 'select';
  required: boolean;
  secret?: boolean;
  placeholder?: string;
  options?: Array<{ value: string; label: string }>;
};

export type DatabaseCatalogEntry = {
  type: string;
  label: string;
  logoUrl: string;
  sourceUrl: string;
  defaults: {
    host: string;
    port: number;
    database: string;
    username: string;
    ssl_enabled: boolean;
    extra: Record<string, string | boolean | number>;
  };
  extraFields: DatabaseExtraField[];
};

// This shared manifest is also validated by the backend connector factory.
// Keep product support, defaults, and type-specific form fields in one place.
export const DATABASE_CATALOG = rawCatalog as DatabaseCatalogEntry[];
export const DATABASE_TYPES = DATABASE_CATALOG.map(({ type }) => type);

const DATABASE_CATALOG_BY_TYPE = new Map(DATABASE_CATALOG.map((database) => [database.type, database]));

export function databaseCatalogEntry(type: string) {
  return DATABASE_CATALOG_BY_TYPE.get(type);
}

export function databaseTypeLabel(type: string) {
  return databaseCatalogEntry(type)?.label || type;
}
