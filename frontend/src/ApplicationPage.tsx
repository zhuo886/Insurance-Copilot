import {
  ArrowLeftOutlined,
  AuditOutlined,
  CalculatorOutlined,
  CheckCircleFilled,
  CheckCircleOutlined,
  FileProtectOutlined,
  FileTextOutlined,
  HeartOutlined,
  IdcardOutlined,
  PayCircleOutlined,
  SafetyCertificateOutlined,
  VerifiedOutlined,
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
        <div className="advisor-header__left">
          <div className="brand">
            <span className="brand__mark"><SafetyCertificateOutlined /></span>
            <span>
              <strong>安心保</strong>
              <small>在线投保</small>
            </span>
          </div>
        </div>
        <div className="advisor-header__right">
          <span className="advisor-user-badge">{session.user.displayName || session.user.username}</span>
          <Button icon={<ArrowLeftOutlined />} onClick={onBack}>返回顾问</Button>
        </div>
      </header>

      <main className="application-shell">
        <div className="application-grid">
          <div className="application-main">
            <section className="application-header">
              {loadingMeta ? (
                <Spin />
              ) : (
                <div>
                  <span className="application-header__eyebrow">
                    {product ? `${CATEGORY_LABELS[product.category] ?? product.category} · ${product.insurer}` : '当前产品'}
                  </span>
                  <h1>{product ? product.name : productId ? `产品 ID：${productId}` : '未选择产品'}</h1>
                  <p>{product?.description || '支持在线保费试算、预约下单与方案投保。'}</p>
                </div>
              )}
            </section>

            <section className="application-flow">
              <Steps current={paid ? APPLICATION_STEPS.length : currentStep} items={APPLICATION_STEPS} />
              {error ? <Alert showIcon type="error" message={error} /> : null}

              {paid ? (
                <div className="policy-receipt-card">
                  <div className="policy-receipt-card__head">
                    <div className="policy-receipt-card__seal">
                      <VerifiedOutlined />
                    </div>
                    <div>
                      <h2>投保成功 · 保单已生效</h2>
                    </div>
                  </div>
                  <p className="policy-receipt-card__desc">
                    保单状态已更新为「生效中」，可回到智能顾问查询保单详情及理赔指南。
                  </p>
                  <Descriptions bordered size="small" column={1}>
                    <Descriptions.Item label="生效产品">{product?.name ?? productId ?? '-'}</Descriptions.Item>
                    <Descriptions.Item label="被保人">{session.user.displayName || session.user.username}</Descriptions.Item>
                    <Descriptions.Item label="年缴保费">
                      {quote ? `¥${quote.annual_premium} / 年` : '已按方案配置生效'}
                    </Descriptions.Item>
                    {order ? <Descriptions.Item label="订单号">{order.order_number}</Descriptions.Item> : null}
                  </Descriptions>
                  <div className="policy-receipt-card__actions">
                    <Button type="primary" icon={<ArrowLeftOutlined />} onClick={onBack}>
                      返回智能顾问
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="application-step-body">
                  <div className="step-body-head">
                    <h2>{APPLICATION_STEPS[currentStep].title}</h2>
                  </div>

                  {currentStep === 0 && fields.length > 0 ? (
                    <div>
                      <Form layout="vertical">
                        <div className="pricing-form-grid">
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
                        </div>
                        <Button
                          type="primary"
                          icon={<CalculatorOutlined />}
                          loading={calculating}
                          onClick={() => void runCalculate()}
                        >
                          立即试算保费
                        </Button>
                      </Form>

                      {quote ? (
                        <div className="quote-receipt-card">
                          <div className="quote-receipt-card__top">
                            <div>
                              <span>报价单号：{quote.quote_id}</span>
                              <div className="quote-receipt-card__amount">
                                <strong>¥{quote.annual_premium}</strong>
                                <small>/ 年</small>
                              </div>
                            </div>
                            <Tag color={quote.pricing_mode === 'table' ? 'green' : 'gold'}>
                              {quote.pricing_mode === 'table' ? '官方费率表' : '区间参考'}
                            </Tag>
                          </div>

                          {quote.warnings.length > 0 ? (
                            <ul className="quote-receipt-card__notes">
                              {quote.warnings.map((w, idx) => (
                                <li key={idx}>{w}</li>
                              ))}
                            </ul>
                          ) : null}

                          <div className="quote-receipt-card__footer">
                            {order ? (
                              <Tag color="cyan">
                                已保存待支付订单：{order.order_number}（¥{order.total_premium}）
                              </Tag>
                            ) : (
                              <Button size="small" loading={creatingOrder} onClick={() => void submitReservationOrder()}>
                                保存待支付订单
                              </Button>
                            )}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : currentStep === 0 ? (
                    <div className="disclosure-card">
                      <strong><AuditOutlined /> 费率说明</strong>
                      <p>当前产品暂无需额外填写参数，可参考右侧基准保费并点击「下一步」继续。</p>
                    </div>
                  ) : currentStep === 1 ? (
                    <div className="disclosure-list">
                      <div className="disclosure-card">
                        <strong><FileProtectOutlined /> 承保与条款确认</strong>
                        <p>
                          本产品由 <strong>{product?.insurer ?? '对应保险公司'}</strong> 承保，保障责任、等待期与免赔规则以正式保险合同为准。
                        </p>
                      </div>
                      <div className="disclosure-card">
                        <strong><CheckCircleFilled /> 犹豫期说明</strong>
                        <p>电子保单生效后请及时核对保单信息，犹豫期内可按合同约定申请无息退保。</p>
                      </div>
                    </div>
                  ) : currentStep === 2 ? (
                    <div className="disclosure-list">
                      <div className="disclosure-card">
                        <strong><HeartOutlined /> 健康告知确认</strong>
                        <p>请确认被保人既往住院记录、慢性病史及体检结论符合产品投保健康要求。</p>
                      </div>
                      <div className="disclosure-card">
                        <strong><VerifiedOutlined /> 年龄与职业确认</strong>
                        <p>请确认被保人当前年龄及职业分类在产品可承保范围内。</p>
                      </div>
                    </div>
                  ) : currentStep === 3 ? (
                    <Descriptions bordered size="small" column={1}>
                      <Descriptions.Item label="产品名称">{product?.name ?? productId ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="承保公司">{product?.insurer ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="被保人">{session.user.displayName || session.user.username}</Descriptions.Item>
                      <Descriptions.Item label="试算保费">
                        {quote ? `¥${quote.annual_premium} / 年（${quote.pricing_mode === 'table' ? '官方费率表' : '参考估算'}）` : '沿用方案参考预算'}
                      </Descriptions.Item>
                      {order ? <Descriptions.Item label="关联订单号">{order.order_number}</Descriptions.Item> : null}
                    </Descriptions>
                  ) : (
                    <div className="disclosure-card">
                      <strong><PayCircleOutlined /> 确认支付</strong>
                      <p>核对以上信息无误后，点击「支付并生效保单」完成投保。</p>
                    </div>
                  )}

                  {!planId || !itemId ? (
                    <Alert
                      showIcon
                      type="warning"
                      message="当前仅作试算与投保演示"
                      description="未携带方案项 ID（plan_id / item_id），可正常试算与保存订单，确认支付需从顾问方案卡片进入。"
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
          </div>

          <aside className="application-sidebar" aria-label="报价摘要">
            <div className="sidebar-summary-card">
              <div className="sidebar-summary-card__head">
                <strong>报价摘要</strong>
                <span className="sidebar-status-tag">
                  {paid ? '已生效' : quote ? '已试算' : '待试算'}
                </span>
              </div>

              <div className="sidebar-product-meta">
                <h4>{product?.name ?? (productId ? `产品 ID ${productId}` : '未指定产品')}</h4>
                <p>
                  {product ? `${product.insurer} · ${CATEGORY_LABELS[product.category] ?? product.category}` : '正在加载产品详情'}
                </p>
              </div>

              <div className="sidebar-quote-box">
                <span>{quote ? '本次试算年保费' : '年缴保费参考'}</span>
                <div className="sidebar-quote-box__price">
                  <strong>
                    {quote
                      ? `¥${Number.isFinite(Number(quote.annual_premium)) ? Number(quote.annual_premium).toLocaleString('zh-CN') : quote.annual_premium}`
                      : product
                        ? `¥${product.min_premium.toLocaleString('zh-CN')}`
                        : '--'}
                  </strong>
                  <small>{quote ? '/ 年' : '起 / 年'}</small>
                </div>
                {quote ? (
                  <div className="sidebar-quote-box__meta">
                    <Tag color={quote.pricing_mode === 'table' ? 'green' : 'gold'}>
                      {quote.pricing_mode === 'table' ? '官方费率表' : '区间参考'}
                    </Tag>
                    <small>单号：{quote.quote_id}</small>
                  </div>
                ) : (
                  <small className="sidebar-quote-box__hint">选择左侧参数后点击试算</small>
                )}
              </div>

              {order ? (
                <div className="sidebar-order-chip">
                  <span>预约订单号</span>
                  <strong>{order.order_number}</strong>
                </div>
              ) : null}

              {product?.highlights?.length ? (
                <div className="sidebar-highlights">
                  <h5>保障亮点</h5>
                  <ul>
                    {product.highlights.slice(0, 3).map((item) => (
                      <li key={item}>
                        <CheckCircleFilled />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
