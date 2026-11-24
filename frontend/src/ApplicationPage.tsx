import {
  ArrowLeftOutlined,
  CheckCircleOutlined,
  FileTextOutlined,
  HeartOutlined,
  IdcardOutlined,
  PayCircleOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import { Alert, Button, Descriptions, Form, InputNumber, Select, Spin, Steps, Tag } from 'antd';
import { useEffect, useState } from 'react';

import {
  ApiError,
  calculatePremium,
  createOrder,
  getProductDetail,
  getProductPricingSchema,
  markPlanItemInsured,
} from './api';
import { CATEGORY_LABELS } from './categoryLabels';
import type {
  AuthSession,
  OrderResponse,
  PremiumCalculateResponse,
  PricingSchemaField,
  PricingSchemaResponse,
  Product,
} from './types';

interface ApplicationPageProps {
  session: AuthSession;
  onBack: () => void;
  onSessionExpired: () => void;
}

const APPLICATION_STEPS = [
  { title: '保费试算', icon: <PayCircleOutlined /> },
  { title: '客户告知', icon: <FileTextOutlined /> },
  { title: '健康告知', icon: <HeartOutlined /> },
  { title: '保单信息', icon: <IdcardOutlined /> },
  { title: '确认支付', icon: <CheckCircleOutlined /> },
];

function resolveFieldOptions(
  field: PricingSchemaField,
  schemaData?: PricingSchemaResponse['schema'],
): Array<{ label: string; value: string | number }> {
  const rawList =
    field.options ??
    (field.options_ref && schemaData && Array.isArray(schemaData[field.options_ref])
      ? (schemaData[field.options_ref] as Array<Record<string, unknown>>)
      : []);

  return rawList.map((opt) => {
    if (typeof opt === 'number' || typeof opt === 'string') {
      const label =
        field.type === 'money' && typeof opt === 'number'
          ? `${(opt / 10000).toFixed(opt % 10000 === 0 ? 0 : 1)} 万元`
          : String(opt);
      return { label, value: opt };
    }
    return {
      label: String(opt.label ?? opt.value ?? ''),
      value: (opt.value ?? opt.label) as string | number,
    };
  });
}

function buildInitialAnswers(schemaResp: PricingSchemaResponse): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  for (const field of schemaResp.schema.fields ?? []) {
    const opts = resolveFieldOptions(field, schemaResp.schema);
    if (field.key === 'insured.age') {
      const minA = typeof field.min === 'number' ? field.min : 0;
      const maxA = typeof field.max === 'number' ? field.max : 65;
      defaults[field.key] = Math.min(maxA, Math.max(minA, 30));
    } else if (field.key === 'plan_code') {
      const hasStd = opts.some((o) => o.value === 'standard');
      defaults[field.key] = hasStd ? 'standard' : opts[0]?.value ?? 'standard';
    } else if (opts.length > 0) {
      defaults[field.key] = opts[0].value;
    }
  }
  return defaults;
}

export function ApplicationPage({ session, onBack, onSessionExpired }: ApplicationPageProps) {
  const params = new URLSearchParams(window.location.search);
  const productIdStr = params.get('product_id');
  const productId = productIdStr ? Number(productIdStr) : null;
  const planId = params.get('plan_id');
  const itemId = params.get('item_id');

  const [product, setProduct] = useState<Product | null>(null);
  const [pricingSchema, setPricingSchema] = useState<PricingSchemaResponse | null>(null);
  const [loadingMeta, setLoadingMeta] = useState(Boolean(productId));
  const [answers, setAnswers] = useState<Record<string, unknown>>({});
  const [calculating, setCalculating] = useState(false);
  const [quote, setQuote] = useState<PremiumCalculateResponse | null>(null);
  const [order, setOrder] = useState<OrderResponse | null>(null);
  const [creatingOrder, setCreatingOrder] = useState(false);

  const [currentStep, setCurrentStep] = useState(0);
  const [paying, setPaying] = useState(false);
  const [paid, setPaid] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!productId || Number.isNaN(productId)) {
      setLoadingMeta(false);
      return;
    }
    const controller = new AbortController();
    setLoadingMeta(true);
    setError(null);
    Promise.all([
      getProductDetail(productId, controller.signal),
      getProductPricingSchema(productId, controller.signal),
    ])
      .then(([prod, schemaResp]) => {
        setProduct(prod);
        setPricingSchema(schemaResp);
        setAnswers(buildInitialAnswers(schemaResp));
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return;
        setError(reason instanceof Error ? reason.message : '产品信息加载失败');
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingMeta(false);
      });
    return () => controller.abort();
  }, [productId]);

  async function runCalculate() {
    if (!productId || calculating) return;
    setCalculating(true);
    setError(null);
    try {
      const res = await calculatePremium({
        product_id: productId,
        scenario: pricingSchema?.scenario ?? 'premium_by_coverage',
        answers,
      });
      setQuote(res);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保费试算失败，请检查填写项');
    } finally {
      setCalculating(false);
    }
  }

  async function submitReservationOrder() {
    if (!quote || creatingOrder) return;
    setCreatingOrder(true);
    setError(null);
    try {
      const created = await createOrder(session.user.id, {
        quote_ids: [quote.quote_id],
        idempotency_key: `apply-${quote.quote_id}`,
        recommendation_snapshot: product
          ? { product_id: product.id, product_name: product.name }
          : undefined,
      });
      setOrder(created);
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 401) {
        onSessionExpired();
        return;
      }
      setError(reason instanceof Error ? reason.message : '创建投保订单失败');
    } finally {
      setCreatingOrder(false);
    }
  }

  async function pay() {
    if (!planId || !itemId || paying || paid) return;
    setPaying(true);
    setError(null);
    try {
      await markPlanItemInsured(session.user.id, planId, itemId);
      setPaid(true);
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 401) {
        onSessionExpired();
        return;
      }
      setError(reason instanceof Error ? reason.message : '支付失败，请稍后重试');
    } finally {
      setPaying(false);
    }
  }

  const isLastStep = currentStep === APPLICATION_STEPS.length - 1;
  const fields = pricingSchema?.schema.fields ?? [];

  return (
    <div className="application-page">
      <header className="advisor-header">
        <div className="brand">
          <span className="brand__mark"><SafetyCertificateOutlined /></span>
          <span><strong>安心保</strong><small>投保与精算试算流程</small></span>
        </div>
        <div>
          <span>{session.user.displayName || session.user.username}</span>
          <Button icon={<ArrowLeftOutlined />} onClick={onBack}>返回顾问</Button>
        </div>
      </header>

      <main className="application-shell">
        <section className="application-header">
          {loadingMeta ? (
            <Spin />
          ) : (
            <div>
              <span>
                {product ? `${CATEGORY_LABELS[product.category] ?? product.category} · ${product.insurer}` : '当前产品'}
              </span>
              <h1>{product ? product.name : productId ? `产品 ID：${productId}` : '未选择产品'}</h1>
              <p>
                {product?.description || '支持基于官方费率表在线保费试算、生单预约与方案投保落库。'}
              </p>
            </div>
          )}
        </section>

        <section className="application-flow">
          <Steps current={paid ? APPLICATION_STEPS.length : currentStep} items={APPLICATION_STEPS} />
          {error ? <Alert showIcon type="error" message={error} /> : null}
          {paid ? (
            <Alert
              showIcon
              type="success"
              message="投保成功"
              description="保单状态已更新为生效中并写入保单表，后续可在智能顾问中查询该保单进度及理赔指引。"
            />
          ) : (
            <div className="application-step-body">
              <h2>{APPLICATION_STEPS[currentStep].title}</h2>

              {currentStep === 0 && fields.length > 0 ? (
                <div>
                  <Form layout="vertical">
                    {fields.map((field) => {
                      const opts = resolveFieldOptions(field, pricingSchema?.schema);
                      if (field.type === 'integer' || field.type === 'number') {
                        return (
                          <Form.Item key={field.key} label={field.label} required={field.required}>
                            <InputNumber
                              style={{ width: '100%' }}
                              min={field.min ?? 0}
                              max={field.max ?? 120}
                              value={typeof answers[field.key] === 'number' ? (answers[field.key] as number) : undefined}
                              onChange={(val) =>
                                setAnswers((prev) => ({ ...prev, [field.key]: val }))
                              }
                            />
                          </Form.Item>
                        );
                      }
                      return (
                        <Form.Item key={field.key} label={field.label} required={field.required}>
                          <Select
                            options={opts}
                            value={answers[field.key] as string | number | undefined}
                            onChange={(val) =>
                              setAnswers((prev) => ({ ...prev, [field.key]: val }))
                            }
                          />
                        </Form.Item>
                      );
                    })}
                    <Button type="primary" loading={calculating} onClick={() => void runCalculate()}>
                      立即试算保费
                    </Button>
                  </Form>

                  {quote ? (
                    <div style={{ marginTop: 16 }}>
                      <Alert
                        showIcon
                        type={quote.pricing_mode === 'table' ? 'success' : 'info'}
                        message={
                          <span>
                            预估年缴保费：<strong>¥{quote.annual_premium}</strong>{' '}
                            <Tag color={quote.pricing_mode === 'table' ? 'green' : 'orange'}>
                              {quote.pricing_mode === 'table' ? '官方精算费率表' : '区间参考估算'}
                            </Tag>
                          </span>
                        }
                        description={
                          <div>
                            <p style={{ marginBottom: 6 }}>报价单号：{quote.quote_id}</p>
                            {quote.warnings.map((w, idx) => (
                              <div key={idx} style={{ fontSize: 12 }}>· {w}</div>
                            ))}
                          </div>
                        }
                      />
                      <div style={{ marginTop: 10 }}>
                        {order ? (
                          <Tag color="blue">已创建待支付订单：{order.order_number}（¥{order.total_premium}）</Tag>
                        ) : (
                          <Button size="small" loading={creatingOrder} onClick={() => void submitReservationOrder()}>
                            保存为待支付订单
                          </Button>
                        )}
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : currentStep === 3 ? (
                <Descriptions bordered size="small" column={1}>
                  <Descriptions.Item label="产品名称">{product?.name ?? productId ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="被保人">{session.user.displayName || session.user.username}</Descriptions.Item>
                  <Descriptions.Item label="试算保费">
                    {quote ? `¥${quote.annual_premium} / 年（${quote.pricing_mode === 'table' ? '官方费率表' : '参考估算'}）` : '未单独试算，沿用方案参考预算'}
                  </Descriptions.Item>
                  {order ? <Descriptions.Item label="关联订单号">{order.order_number}</Descriptions.Item> : null}
                </Descriptions>
              ) : (
                <p>请核对以上条款与告知事项，确认无误后点击「下一步」继续投保流程。</p>
              )}

              {!planId || !itemId ? (
                <Alert
                  showIcon
                  type="warning"
                  message="当前仅作保费试算与投保演示"
                  description="当前入口未携带方案项 ID（plan_id / item_id），可正常完成保费试算与下单，但确认支付更新方案状态需从顾问方案卡片进入。"
                />
              ) : null}

              <div className="application-actions">
                {currentStep > 0 ? (
                  <Button onClick={() => setCurrentStep((step) => step - 1)}>上一步</Button>
                ) : null}
                {isLastStep ? (
                  <Button
                    type="primary"
                    icon={<PayCircleOutlined />}
                    loading={paying}
                    disabled={!planId || !itemId}
                    onClick={() => void pay()}
                  >
                    支付并生效保单
                  </Button>
                ) : (
                  <Button type="primary" onClick={() => setCurrentStep((step) => step + 1)}>
                    下一步
                  </Button>
                )}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
