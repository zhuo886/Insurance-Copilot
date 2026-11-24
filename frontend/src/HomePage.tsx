import {
  ArrowRightOutlined,
  AuditOutlined,
  CheckCircleFilled,
  CustomerServiceOutlined,
  LoginOutlined,
  LogoutOutlined,
  SafetyCertificateOutlined,
  VerifiedOutlined,
} from '@ant-design/icons';
import { Alert, Button, Empty, Modal, Segmented, Skeleton } from 'antd';
import { useEffect, useState } from 'react';

import { listProducts } from './api';
import { AuthModal } from './AuthModal';
import { CATEGORY_LABELS } from './categoryLabels';
import { ProductCard } from './ProductCard';
import type { AuthSession, InsuranceCategory, Product } from './types';

type CategoryFilter = InsuranceCategory | 'all';

const CATEGORY_OPTIONS: Array<{ label: string; value: CategoryFilter }> = [
  { label: '全部精选', value: 'all' },
  { label: '医疗险', value: 'medical' },
  { label: '重疾险', value: 'critical_illness' },
  { label: '终身寿险', value: 'life' },
  { label: '意外险', value: 'accident' },
];

interface HomePageProps {
  session: AuthSession | null;
  initialAuthOpen?: boolean;
  onAuthenticated: (session: AuthSession) => void;
  onLogout: () => void;
  onConsult: () => void;
}

export function HomePage({
  session,
  initialAuthOpen = false,
  onAuthenticated,
  onLogout,
  onConsult,
}: HomePageProps) {
  const [category, setCategory] = useState<CategoryFilter>('all');
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [authOpen, setAuthOpen] = useState(initialAuthOpen);
  const [consultAfterLogin, setConsultAfterLogin] = useState(initialAuthOpen);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    listProducts(category, controller.signal)
      .then(setProducts)
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return;
        setError(reason instanceof Error ? reason.message : '产品加载失败');
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [category, reloadKey]);

  function openAuth(continueToAdvisor = false) {
    setConsultAfterLogin(continueToAdvisor);
    setAuthOpen(true);
  }

  function consult() {
    if (session) {
      onConsult();
      return;
    }
    openAuth(true);
  }

  function authenticated(nextSession: AuthSession) {
    onAuthenticated(nextSession);
    setAuthOpen(false);
    if (consultAfterLogin) onConsult();
    setConsultAfterLogin(false);
  }

  return (
    <div className="page-shell">
      <header className="site-header">
        <div className="site-header__inner">
          <a className="brand" href="/" aria-label="安心保首页">
            <span className="brand__mark"><SafetyCertificateOutlined /></span>
            <span>
              <strong>安心保</strong>
              <small>PRIVATE INSURANCE OFFICE</small>
            </span>
          </a>
          <nav className="site-nav" aria-label="主导航">
            <a className="is-active" href="#products">精选产品</a>
            <a href="#service">私享保障准则</a>
          </nav>
          <div className="site-actions">
            {session ? (
              <>
                <span className="user-pill">
                  {session.user.displayName || session.user.username}
                </span>
                <Button type="text" icon={<LogoutOutlined />} onClick={onLogout}>退出</Button>
              </>
            ) : (
              <Button type="text" icon={<LoginOutlined />} onClick={() => openAuth()}>
                登录 / 注册
              </Button>
            )}
            <Button type="primary" icon={<CustomerServiceOutlined />} onClick={consult}>
              AI 专属顾问
            </Button>
          </div>
        </div>
      </header>

      <main>
        <section className="hero-editorial">
          <div className="hero-editorial__inner">
            <div className="hero-copy">
              <span className="hero-eyebrow">
                12 款真实条款 · 11/12 官方精算费率 · 确定性合规护栏
              </span>
              <h1>
                以精算与条款为据，
                <br />
                定制从容的家庭保障
              </h1>
              <p>
                甄选在售商业保险真实合同，基于官方精算费率表精准试算保费，
                结合跨产品并行条款检索与承保红线硬校验，让每一项保障选择都经得起原文推敲。
              </p>
              <div className="hero-cta-group">
                <Button
                  type="primary"
                  size="large"
                  icon={<CustomerServiceOutlined />}
                  onClick={consult}
                >
                  开启专属保障规划
                </Button>
                <a className="hero-secondary-link" href="#products">
                  浏览精选保障 <ArrowRightOutlined />
                </a>
              </div>
            </div>

            <aside className="hero-showcase" aria-label="私行保障配置示例">
              <div className="hero-showcase__head">
                <div>
                  <span>家庭保障组合诊断示例</span>
                  <strong>30 岁有医保 · 均衡防线配置</strong>
                </div>
                <span className="hero-showcase__badge">
                  <AuditOutlined /> 官方费率核验
                </span>
              </div>

              <div className="hero-showcase__rows">
                <div className="showcase-row">
                  <div>
                    <strong>百万医疗险防线</strong>
                    <small>住院与特药报销 · 20 年保证续保</small>
                  </div>
                  <div className="showcase-row__right">
                    <span>¥290–¥360/年</span>
                    <div className="showcase-bar"><i style={{ width: '44%' }} /></div>
                  </div>
                </div>
                <div className="showcase-row">
                  <div>
                    <strong>核心重疾给付防线</strong>
                    <small>收入损失补偿 · 官方费率查表试算</small>
                  </div>
                  <div className="showcase-row__right">
                    <span>¥2,500–¥2,900/年</span>
                    <div className="showcase-bar"><i style={{ width: '84%' }} /></div>
                  </div>
                </div>
                <div className="showcase-row">
                  <div>
                    <strong>综合意外及猝死防线</strong>
                    <small>全天候交通意外与意外医疗赔付</small>
                  </div>
                  <div className="showcase-row__right">
                    <span>¥180–¥296/年</span>
                    <div className="showcase-bar"><i style={{ width: '28%' }} /></div>
                  </div>
                </div>
              </div>

              <div className="hero-showcase__checks">
                <span><CheckCircleFilled /> 条款原文 [ref] 溯源</span>
                <span><VerifiedOutlined /> 年龄/职业红线零越界</span>
              </div>

              <button type="button" className="hero-showcase__action" onClick={consult}>
                <span>立即为我试算专属组合</span>
                <ArrowRightOutlined />
              </button>
            </aside>
          </div>
        </section>

        <section className="products-section" id="products">
          <div className="products-heading">
            <div>
              <span className="section-kicker">CURATED PORTFOLIO</span>
              <h2>真实合同精选保障库</h2>
              <p>每款产品均对应真实备案合同条款，支持顾问逐句检索比对与在线保费试算</p>
            </div>
            <span className="products-count">
              {loading ? '正在整理产品目录...' : `当前展示 ${products.length} 款在售产品`}
            </span>
          </div>

          <div className="category-filter" aria-label="按保险类型筛选">
            <Segmented<CategoryFilter>
              value={category}
              options={CATEGORY_OPTIONS}
              onChange={setCategory}
            />
          </div>

          {error ? (
            <Alert
              className="products-alert"
              type="error"
              showIcon
              message="产品加载失败"
              description={error}
              action={<Button onClick={() => setReloadKey((value) => value + 1)}>重新加载</Button>}
            />
          ) : null}

          {loading ? (
            <div className="product-grid" aria-label="产品加载中">
              {Array.from({ length: 6 }, (_, index) => (
                <div className="product-skeleton" key={index}><Skeleton active /></div>
              ))}
            </div>
          ) : products.length ? (
            <div className="product-grid">
              {products.map((product) => (
                <ProductCard key={product.id} product={product} onView={setSelectedProduct} />
              ))}
            </div>
          ) : (
            <Empty description="该类型暂时没有可展示的产品" />
          )}
        </section>

        <section className="service-strip" id="service">
          <article className="trust-card">
            <span className="trust-card__num">01</span>
            <strong>合同条款逐句溯源</strong>
            <p>跨产品并行检索备案原文，每项关键数字均标注合法来源编号，拒绝模板兜底与无依据演绎。</p>
          </article>
          <article className="trust-card">
            <span className="trust-card__num">02</span>
            <strong>官方费率精算试算</strong>
            <p>结构化接入官方精算费率表，按年龄、性别、社保与缴费期精准试算；缺表项明示区间估算。</p>
          </article>
          <article className="trust-card">
            <span className="trust-card__num">03</span>
            <strong>承保红线确定性护栏</strong>
            <p>硬核校验被保人年龄上限、职业类别与家庭年度预算，并需经您确认审批后方可落库生单。</p>
          </article>
        </section>
      </main>

      <footer className="site-footer">
        保险产品信息仅供参考，具体保障责任、等待期与免赔规定以对应保险合同条款及正式投保页面为准。
      </footer>

      <AuthModal
        open={authOpen}
        onCancel={() => {
          setAuthOpen(false);
          setConsultAfterLogin(false);
        }}
        onSuccess={authenticated}
      />

      <Modal
        open={selectedProduct !== null}
        title={selectedProduct?.name}
        footer={[
          <Button key="close" onClick={() => setSelectedProduct(null)}>关闭</Button>,
          <Button key="consult" type="primary" onClick={consult}>向顾问咨询此产品</Button>,
        ]}
        onCancel={() => setSelectedProduct(null)}
        width={640}
      >
        {selectedProduct ? (
          <div className="product-detail">
            <p>{selectedProduct.description}</p>
            <dl>
              <div><dt>承保公司</dt><dd>{selectedProduct.insurer}</dd></div>
              <div><dt>保险类型</dt><dd>{CATEGORY_LABELS[selectedProduct.category]}</dd></div>
              <div><dt>适合人群</dt><dd>{selectedProduct.target_group}</dd></div>
              <div><dt>参考起步价</dt><dd>¥{selectedProduct.min_premium.toLocaleString('zh-CN')} 起 / 年</dd></div>
            </dl>
            <h4>核心保障亮点</h4>
            <ul>{selectedProduct.highlights.map((item) => <li key={item}>{item}</li>)}</ul>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
