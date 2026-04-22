// HTTP client for the shared adam-billing service. Mirrors
// onshape-extension/src/lib/billing/client.ts so CADAM and onshape behave
// identically against the same endpoints.

export type SubscriptionLevel = 'standard' | 'pro';

export type BillingStatus = {
  user: {
    hasTrialed: boolean;
  };
  subscription: {
    level: SubscriptionLevel;
    status: string | null;
    currentPeriodEnd: string | null;
  } | null;
  tokens: {
    free: number;
    subscription: number;
    purchased: number;
    total: number;
  };
};

export type ConsumeSuccess = {
  ok: true;
  tokensDeducted: number;
  freeBalance: number;
  subscriptionBalance: number;
  purchasedBalance: number;
  totalBalance: number;
};

export type ConsumeFailure = {
  ok: false;
  reason: 'insufficient_tokens';
  tokensRequired: number;
  tokensAvailable: number;
  tokensDeducted: number;
};

export type ConsumeResult = ConsumeSuccess | ConsumeFailure;

export type RefundResult = {
  ok: true;
  tokensRefunded: number;
  source: 'subscription' | 'purchased';
  freeBalance: number;
  subscriptionBalance: number;
  purchasedBalance: number;
  totalBalance: number;
};

export type BillingProduct = {
  id: string;
  stripeProductId: string;
  stripePriceId: string;
  productType: 'subscription' | 'pack';
  subscriptionLevel: SubscriptionLevel | null;
  tokenAmount: number;
  name: string;
  priceCents: number;
  interval: string | null;
  active: boolean;
};

export class BillingClientError extends Error {
  readonly status: number;
  readonly body: unknown;
  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

// Mock billing responses for local development
function mockBillingResponse(
  method: 'GET' | 'POST',
  path: string,
  body?: unknown,
): unknown {
  // Mock consume response - always succeed with unlimited tokens
  if (method === 'POST' && path.includes('/consume')) {
    const consumeBody = body as ConsumeBody;
    return {
      ok: true,
      tokensDeducted: consumeBody?.tokens ?? 0,
      freeBalance: 999999,
      subscriptionBalance: 0,
      purchasedBalance: 0,
      totalBalance: 999999,
    } as ConsumeSuccess;
  }

  // Mock status response - unlimited tokens
  if (method === 'GET' && path.includes('/status')) {
    return {
      user: {
        hasTrialed: false,
      },
      subscription: null,
      tokens: {
        free: 999999,
        subscription: 0,
        purchased: 0,
        total: 999999,
      },
    } as BillingStatus;
  }

  // Mock refund response
  if (method === 'POST' && path.includes('/refund')) {
    const refundBody = body as RefundBody;
    return {
      ok: true,
      tokensRefunded: refundBody?.tokens ?? 0,
      source: 'subscription',
      freeBalance: 999999,
      subscriptionBalance: 0,
      purchasedBalance: 0,
      totalBalance: 999999,
    } as RefundResult;
  }

  // Mock checkout/portal/cancel - return empty success
  if (method === 'POST') {
    if (path.includes('/checkout') || path.includes('/portal')) {
      return { url: 'http://localhost:3000/cadam' };
    }
    if (path.includes('/cancel-subscription')) {
      return { canceled: false, reason: 'no_subscription' };
    }
  }

  // Mock products
  if (method === 'GET' && path.includes('/products')) {
    return path.includes('?type=')
      ? []
      : { subscriptions: [], packs: [] };
  }

  // Default empty response
  return {};
}

const baseUrl = (): string | null => {
  const url = Deno.env.get('BILLING_SERVICE_URL');
  if (!url) return null; // Local dev mode - no billing service
  return url.replace(/\/$/, '');
};

const apiKey = (): string | null => {
  const key = Deno.env.get('BILLING_SERVICE_KEY');
  if (!key) return null; // Local dev mode - no billing service
  return key;
};

type CallOptions = {
  allowStatus?: number[];
};

const call = async <T>(
  method: 'GET' | 'POST',
  path: string,
  body?: unknown,
  options?: CallOptions,
): Promise<T> => {
  const base = baseUrl();
  const key = apiKey();

  // Local dev mode - return mock responses
  if (!base || !key) {
    console.log(`[LOCAL DEV] Skipping billing ${method} ${path}`);
    return mockBillingResponse(method, path, body) as T;
  }

  const res = await fetch(`${base}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let parsed: unknown;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  if (!res.ok && !options?.allowStatus?.includes(res.status)) {
    throw new BillingClientError(
      `billing ${method} ${path} -> ${res.status}`,
      res.status,
      parsed,
    );
  }
  return parsed as T;
};

const enc = (email: string): string => encodeURIComponent(email.toLowerCase());

type ConsumeBody = {
  tokens: number;
  operation?: string;
  referenceId?: string;
};

type RefundBody = {
  tokens: number;
  operation?: string;
  referenceId?: string;
};

type CheckoutBody = {
  priceId: string;
  successUrl: string;
  cancelUrl: string;
  trialPeriodDays?: number;
};

type CancelSubscriptionBody = {
  feedback?:
    | 'customer_service'
    | 'low_quality'
    | 'missing_features'
    | 'other'
    | 'switched_service'
    | 'too_complex'
    | 'too_expensive'
    | 'unused';
  comment?: string;
};

export type CancelSubscriptionResult =
  | { canceled: true }
  | { canceled: false; reason: 'no_subscription' | 'already_canceled' };

export const billing = {
  getStatus: (email: string) =>
    call<BillingStatus>('GET', `/v1/users/${enc(email)}/status`),

  consume: (email: string, body: ConsumeBody) =>
    call<ConsumeResult>('POST', `/v1/users/${enc(email)}/consume`, body, {
      allowStatus: [422],
    }),

  refund: (email: string, body: RefundBody) =>
    call<RefundResult>('POST', `/v1/users/${enc(email)}/refund`, body),

  createCheckout: (email: string, body: CheckoutBody) =>
    call<{ url: string }>('POST', `/v1/users/${enc(email)}/checkout`, body),

  createPortal: (email: string, body: { returnUrl: string }) =>
    call<{ url: string }>('POST', `/v1/users/${enc(email)}/portal`, body),

  cancelSubscription: (email: string, body: CancelSubscriptionBody = {}) =>
    call<CancelSubscriptionResult>(
      'POST',
      `/v1/users/${enc(email)}/cancel-subscription`,
      body,
    ),

  getProductsByType: (type: 'subscription' | 'pack') =>
    call<BillingProduct[]>('GET', `/v1/products?type=${type}`),

  getAllProducts: () =>
    call<{ subscriptions: BillingProduct[]; packs: BillingProduct[] }>(
      'GET',
      '/v1/products',
    ),
};
