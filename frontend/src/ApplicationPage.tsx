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
              <small>ACTUARIAL & APPLICATION</small>
            </span>
          </div>
          {product ? (
            <>
              <span className="advisor-header__divider" aria-hidden="true" />
              <div className="advisor-header__thread">
                <AuditOutlined />
                <span>{CATEGORY_LABELS[product.category] ?? product.category}精算投保</span>
              </div>
            </>
          ) : null}
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
                    {product ? `${CATEGORY_LABELS[product.category] ?? product.category} · ${product.insurer}` : '当前投保产品'}
                  </span>
                  <h1>{product ? product.name : productId ? `产品 ID：${productId}` : '未选择产品'}</h1>
                  <p>
                    {product?.description || '支持基于官方精算费率表在线保费试算、生单预约与方案投保落库。'}
                  </p>
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
                      <span>ELECTRONIC POLICY ISSUED</span>
                      <h2>投保成功 · 电子保单已生效</h2>
                    </div>
                  </div>
                  <p className="policy-receipt-card__desc">
                    保单状态已更新为「生效中」并写入用户保单档案表，您可随时回到智能顾问工作台查询该保单明细或获取理赔材料指南。
                  </p>
                  <Descriptions bordered size="small" column={1}>
                    <Descriptions.Item label="生效产品">{product?.name ?? productId ?? '-'}</Descriptions.Item>
                    <Descriptions.Item label="被保人">{session.user.displayName || session.user.username}</Descriptions.Item>
                    <Descriptions.Item label="年缴保费">
                      {quote ? `¥${quote.annual_premium} / 年（${quote.pricing_mode === 'table' ? '官方精算费率表' : '区间参考估算'}）` : '已按方案配置落库'}
                    </Descriptions.Item>
                    {order ? <Descriptions.Item label="投保订单号">{order.order_number}</Descriptions.Item> : null}
                  </Descriptions>
                  <div className="policy-receipt-card__actions">
                    <Button type="primary" icon={<ArrowLeftOutlined />} onClick={onBack}>
                      返回智能顾问查看保单
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="application-step-body">
                  <div className="step-body-head">
                    <span className="step-body-head__num">STEP 0{currentStep + 1}</span>
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
                              <span>精算报价回执 · {quote.quote_id}</span>
                              <div className="quote-receipt-card__amount">
                                <strong>¥{quote.annual_premium}</strong>
                                <small>/ 年</small>
                              </div>
                            </div>
                            <Tag color={quote.pricing_mode === 'table' ? 'green' : 'gold'}>
                              {quote.pricing_mode === 'table' ? '官方精算费率表' : '区间参考估算'}
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
                                已创建待支付预约单：{order.order_number}（¥{order.total_premium}）
                              </Tag>
                            ) : (
                              <Button size="small" loading={creatingOrder} onClick={() => void submitReservationOrder()}>
                                保存为待支付预约单
                              </Button>
                            )}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : currentStep === 0 ? (
                    <div className="disclosure-card">
                      <strong><AuditOutlined /> 当前产品参考费率说明</strong>
                      <p>暂未加载到动态试算表单字段，您可参考右侧产品基准起步价并继续确认投保告知流程。</p>
                    </div>
                  ) : currentStep === 1 ? (
                    <div className="disclosure-list">
                      <div className="disclosure-card">
                        <strong><FileProtectOutlined /> 承保主体与核心条款核对</strong>
                        <p>
                          本保障产品由 <strong>{product?.insurer ?? '对应备案保险公司'}</strong> 承保，保障责任、等待期天数、免赔额及免除责任均以正式备案条款为准。
                        </p>
                      </div>
                      <div className="disclosure-card">
                        <strong><CheckCircleFilled /> 犹豫期与电子保单送达</strong>
                        <p>投保生效后将生成电子保单快照，请您在收到合同回执后核对被保人信息及犹豫期权益。</p>
                      </div>
                    </div>
                  ) : currentStep === 2 ? (
                    <div className="disclosure-list">
                      <div className="disclosure-card">
                        <strong><HeartOutlined /> 健康状况如实告知提示</strong>
                        <p>投保前请确认被保人既往住院记录、慢性病史及体检异常情况均符合产品健康告知要求，避免影响后续理赔认定。</p>
                      </div>
                      <div className="disclosure-card">
                        <strong><VerifiedOutlined /> 年龄与职业类别合规核验</strong>
                        <p>请确认被保人当前年龄与真实职业分类在承保允许范围内（部分产品不承保 5–6 类高危职业）。</p>
                      </div>
                    </div>
                  ) : currentStep === 3 ? (
                    <Descriptions bordered size="small" column={1}>
                      <Descriptions.Item label="产品名称">{product?.name ?? productId ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="承保公司">{product?.insurer ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="被保人">{session.user.displayName || session.user.username}</Descriptions.Item>
                      <Descriptions.Item label="试算保费">
                        {quote ? `¥${quote.annual_premium} / 年（${quote.pricing_mode === 'table' ? '官方费率表' : '参考估算'}）` : '未单独试算，沿用方案参考预算'}
                      </Descriptions.Item>
                      {order ? <Descriptions.Item label="关联订单号">{order.order_number}</Descriptions.Item> : null}
                    </Descriptions>
                  ) : (
                    <div className="disclosure-card">
                      <strong><PayCircleOutlined /> 确认支付并生成生效保单</strong>
                      <p>点击下方「支付并生效保单」后，系统将同步更新方案保障状态并落库生成您的有效保单。</p>
                    </div>
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
          </div>

          <aside className="application-sidebar" aria-label="精算档案与报价摘要">
            <div className="sidebar-summary-card">
              <div className="sidebar-summary-card__head">
                <div>
                  <span>ACTUARIAL BRIEF</span>
                  <strong>精算档案与报价摘要</strong>
                </div>
                <span className="sidebar-status-tag">
                  {paid ? '已承保' : quote ? '已试算' : '待试算'}
                </span>
              </div>

              <div className="sidebar-product-meta">
                <h4>{product?.name ?? (productId ? `产品 ID ${productId}` : '未指定产品')}</h4>
                <p>
                  {product ? `${product.insurer} · ${CATEGORY_LABELS[product.category] ?? product.category}` : '正在读取产品详情'}
                </p>
              </div>

              <div className="sidebar-quote-box">
                <span>{quote ? '本次精算年缴保费' : '产品基准起步参考价'}</span>
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
                      {quote.pricing_mode === 'table' ? '官方精算费率表' : '区间参考估算'}
                    </Tag>
                    <small>单号：{quote.quote_id}</small>
                  </div>
                ) : (
                  <small className="sidebar-quote-box__hint">填写左侧参数后点击试算获取精准年保费</small>
                )}
              </div>

              {order ? (
                <div className="sidebar-order-chip">
                  <span>预约订单编号</span>
                  <strong>{order.order_number}</strong>
                </div>
              ) : null}

              {product?.highlights?.length ? (
                <div className="sidebar-highlights">
                  <h5><FileProtectOutlined /> 核心保障摘要</h5>
                  <ul>
                    {product.highlights.slice(0, 4).map((item) => (
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
