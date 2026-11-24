import {
  ArrowRightOutlined,
  AuditOutlined,
  CalculatorOutlined,
  CheckCircleFilled,
  CustomerServiceOutlined,
  FileProtectOutlined,
  LoginOutlined,
  LogoutOutlined,
  SafetyCertificateOutlined,
  SearchOutlined,
  SortAscendingOutlined,
  VerifiedOutlined,
} from '@ant-design/icons';
import { Alert, Button, Empty, Input, Modal, Segmented, Select, Skeleton } from 'antd';
import { useEffect, useMemo, useState } from 'react';

import { listProducts } from './api';
import { AuthModal } from './AuthModal';
import { CATEGORY_LABELS } from './categoryLabels';
import { ProductCard } from './ProductCard';
import type { AuthSession, InsuranceCategory, Product } from './types';

type CategoryFilter = InsuranceCategory | 'all';
type SortMode = 'default' | 'price_asc' | 'price_desc';
type PendingAuthAction =
  | { type: 'consult' }
  | { type: 'apply'; productId: number }
  | null;

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
  onTryCalculate: (productId: number) => void;
}

export function HomePage({
  session,
  initialAuthOpen = false,
  onAuthenticated,
  onLogout,
  onConsult,
  onTryCalculate,
}: HomePageProps) {
  const [category, setCategory] = useState<CategoryFilter>('all');
  const [searchKeyword, setSearchKeyword] = useState('');
  const [sortMode, setSortMode] = useState<SortMode>('default');
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [authOpen, setAuthOpen] = useState(initialAuthOpen);
  const [pendingAction, setPendingAction] = useState<PendingAuthAction>(
    initialAuthOpen ? { type: 'consult' } : null,
  );
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

  const displayedProducts = useMemo(() => {
    const keyword = searchKeyword.trim().toLowerCase();
    const filtered = keyword
      ? products.filter((item) => {
          const haystack = [
            item.name,
            item.insurer,
            item.description,
            item.target_group,
            ...item.highlights,
          ]
            .join(' ')
            .toLowerCase();
          return haystack.includes(keyword);
        })
      : products;

    if (sortMode === 'default') return filtered;
    const sorted = [...filtered];
    sorted.sort((a, b) =>
      sortMode === 'price_asc'
        ? Number(a.min_premium) - Number(b.min_premium)
        : Number(b.min_premium) - Number(a.min_premium),
    );
    return sorted;
  }, [products, searchKeyword, sortMode]);

  function openAuth(nextAction: PendingAuthAction = null) {
    setPendingAction(nextAction);
    setAuthOpen(true);
  }

  function consult() {
    if (session) {
      setSelectedProduct(null);
      onConsult();
      return;
    }
    openAuth({ type: 'consult' });
  }

  function tryCalculate(productId: number) {
    if (session) {
      setSelectedProduct(null);
      onTryCalculate(productId);
      return;
    }
    openAuth({ type: 'apply', productId });
  }

  function authenticated(nextSession: AuthSession) {
    onAuthenticated(nextSession);
    setAuthOpen(false);
    const action = pendingAction;
    setPendingAction(null);
    if (!action) return;
    setSelectedProduct(null);
    if (action.type === 'consult') {
      onConsult();
    } else if (action.type === 'apply') {
      onTryCalculate(action.productId);
    }
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
            <div className="hero-copy hero-animate">
              <span className="hero-eyebrow">
                私享保险规划 · 备案合同逐句溯源 · 官方精算费率查表
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

              <div className="hero-stat-strip" aria-label="核心系统保障指标">
                <div className="hero-stat-pill">
                  <strong>12 款</strong>
                  <span>真实备案合同</span>
                </div>
                <div className="hero-stat-pill">
                  <strong>11 / 12</strong>
                  <span>官方精算费率表</span>
                </div>
                <div className="hero-stat-pill">
                  <strong>100%</strong>
                  <span>承保红线硬校验</span>
                </div>
              </div>

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

            <aside className="hero-showcase hero-animate hero-animate--delay" aria-label="私行保障配置示例">
              <div className="hero-showcase__head">
                <div>
                  <span>私享家庭保障诊断书</span>
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
              {loading
                ? '正在整理产品目录...'
                : searchKeyword.trim()
                  ? `筛选命中 ${displayedProducts.length} / ${products.length} 款`
                  : `当前展示 ${displayedProducts.length} 款在售产品`}
            </span>
          </div>

          <div className="catalog-toolbar" aria-label="筛选与检索产品">
            <div className="category-filter">
              <Segmented<CategoryFilter>
                value={category}
                options={CATEGORY_OPTIONS}
                onChange={setCategory}
              />
            </div>
            <div className="catalog-toolbar__controls">
              <Input
                allowClear
                className="catalog-search"
                prefix={<SearchOutlined />}
                placeholder="搜索产品名称、保司、亮点..."
                value={searchKeyword}
                onChange={(event) => setSearchKeyword(event.target.value)}
              />
              <Select<SortMode>
                className="catalog-sort"
                value={sortMode}
                onChange={setSortMode}
                suffixIcon={<SortAscendingOutlined />}
                options={[
                  { label: '默认排序', value: 'default' },
                  { label: '年保费从低到高', value: 'price_asc' },
                  { label: '年保费从高到低', value: 'price_desc' },
                ]}
              />
            </div>
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
          ) : displayedProducts.length ? (
            <div className="product-grid">
              {displayedProducts.map((product) => (
                <ProductCard key={product.id} product={product} onView={setSelectedProduct} />
              ))}
            </div>
          ) : (
            <Empty
              description={
                searchKeyword.trim()
                  ? `未找到与“${searchKeyword.trim()}”相关的保险产品`
                  : '该类型暂时没有可展示的产品'
              }
            >
              {searchKeyword.trim() ? (
                <Button onClick={() => setSearchKeyword('')}>清空搜索条件</Button>
              ) : null}
            </Empty>
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
          setPendingAction(null);
        }}
        onSuccess={authenticated}
      />

      <Modal
        open={selectedProduct !== null}
        title={null}
        footer={
          selectedProduct
            ? [
                <Button key="close" onClick={() => setSelectedProduct(null)}>关闭</Button>,
                <Button
                  key="calculate"
                  icon={<CalculatorOutlined />}
                  onClick={() => tryCalculate(selectedProduct.id)}
                >
                  在线试算保费
                </Button>,
                <Button
                  key="consult"
                  type="primary"
                  icon={<CustomerServiceOutlined />}
                  onClick={consult}
                >
                  向顾问咨询此产品
                </Button>,
              ]
            : null
        }
        onCancel={() => setSelectedProduct(null)}
        width={680}
      >
        {selectedProduct ? (
          <div className="product-detail">
            <div className="product-detail__banner">
              <div className="product-detail__banner-copy">
                <div className="product-detail__tags">
                  <span className="detail-category-badge">
                    {CATEGORY_LABELS[selectedProduct.category]}
                  </span>
                  <span className="detail-insurer-badge">{selectedProduct.insurer}</span>
                </div>
                <h3>{selectedProduct.name}</h3>
              </div>
              <div className="product-detail__price-box">
                <span>年缴参考起步</span>
                <div>
                  <strong>¥{selectedProduct.min_premium.toLocaleString('zh-CN')}</strong>
                  <small>起 / 年</small>
                </div>
              </div>
            </div>

            <p className="product-detail__desc">{selectedProduct.description}</p>

            <dl>
              <div><dt>承保公司</dt><dd>{selectedProduct.insurer}</dd></div>
              <div><dt>保险类型</dt><dd>{CATEGORY_LABELS[selectedProduct.category]}</dd></div>
              <div><dt>适合人群</dt><dd>{selectedProduct.target_group}</dd></div>
              <div><dt>保费试算支持</dt><dd>支持在线保费试算与顾问条款解读</dd></div>
            </dl>

            <h4><FileProtectOutlined /> 核心保障亮点</h4>
            <ul>
              {selectedProduct.highlights.map((item) => (
                <li key={item}>
                  <CheckCircleFilled />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
