export type InsuranceCategory = 'medical' | 'critical_illness' | 'life' | 'accident';

export interface Product {
  id: number;
  name: string;
  clause_name: string;
  category: InsuranceCategory;
  insurer: string;
  image_url: string | null;
  description: string;
  min_premium: number;
  max_premium: number | null;
  target_group: string;
  highlights: string[];
  status: string;
}

export interface UserSummary {
  id: number;
  username: string;
  email: string;
  displayName: string | null;
}

export interface AuthSession {
  user: UserSummary;
  accessToken?: string;
}

export interface AuthTokenPayload {
  access_token: string;
  token_type: string;
  user: {
    id: number;
    username: string;
    email: string;
    display_name?: string | null;
    displayName?: string | null;
  };
}

export interface PricingSchemaOption {
  value: string | number;
  label: string;
  description?: string;
}

export interface PricingSchemaField {
  key: string;
  label: string;
  type: string;
  required?: boolean;
  min?: number;
  max?: number;
  options?: Array<PricingSchemaOption | number | string>;
  options_ref?: string;
}

export interface PricingSchemaResponse {
  product_id: number;
  scenario: string;
  schema_version: string;
  schema: {
    title?: string;
    fields?: PricingSchemaField[];
    plan_options?: PricingSchemaOption[];
    [key: string]: unknown;
  };
}

export interface PremiumCalculateResponse {
  quote_id: string;
  product_id: number;
  plan_code: string;
  pricing_mode: string;
  annual_premium: string;
  payment_frequency: string;
  breakdown: Array<{
    item: string;
    name: string;
    premium: string;
    source: Record<string, unknown>;
  }>;
  warnings: string[];
  expires_at: string;
}

export interface OrderResponse {
  id: string;
  user_id: number;
  order_number: string;
  total_premium: string;
  status: string;
  items: Array<Record<string, unknown>>;
  quote_ids: string[];
  created_at: string;
}

export interface ChatThread {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  additionalInfo?: AdditionalInfo[];
  additional_info?: unknown;
}

export interface ChatHistory {
  thread_id: string;
  messages: ChatMessage[];
  interrupt: PlanInterrupt | null;
  additional_info?: unknown;
}

export interface PlanInterrupt {
  plan: unknown;
  confirm_info: string;
  action: Array<'approve' | 'reject'>;
}

export interface ClauseSource {
  source_id: string;
  clause_name?: string | null;
  document_name?: string | null;
  section_path?: string | string[] | null;
  clause_no?: string | null;
  content?: string | null;
  score?: number | null;
  [key: string]: unknown;
}

export interface AdditionalProduct {
  id?: number | string;
  plan_id?: string | null;
  product_id?: number | string;
  name?: string;
  product_name?: string;
  category?: string;
  image_url?: string | null;
  description?: string | null;
  min_premium?: number | string | null;
  annual_premium_budget?: number | string | null;
  target_group?: string | null;
  highlights?: string[];
  recommendation_reason?: string | null;
  reason?: string | null;
  status?: string | null;
  [key: string]: unknown;
}

export interface AdditionalPolicy {
  id?: string;
  policy_number?: string | null;
  application_no?: string | null;
  status?: string | null;
  effective_at?: string | null;
  expires_at?: string | null;
  holder_name?: string | null;
  insured_name?: string | null;
  insured_id_no_masked?: string | null;
  insured_phone?: string | null;
  coverage_amount?: number | string | null;
  premium_amount?: number | string | null;
  payment_frequency?: string | null;
  product_id?: number | string;
  product?: AdditionalProduct;
  coverage_snapshot?: Record<string, unknown> | null;
  policy_snapshot?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export type AdditionalInfo =
  | { type: 'clause_sources'; sources: ClauseSource[] }
  | { type: 'products'; title: string; products: AdditionalProduct[] }
  | { type: 'policies'; title: string; policies: AdditionalPolicy[] };
