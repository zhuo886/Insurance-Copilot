import {
  ArrowRightOutlined,
  CalculatorOutlined,
  CheckCircleFilled,
  CustomerServiceOutlined,
  LoginOutlined,
  LogoutOutlined,
  SafetyCertificateOutlined,
  SearchOutlined,
  SortAscendingOutlined,
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
  { label: '全部产品', value: 'all' },
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
              <small>保险优选</small>
            </span>
          </a>
          <nav className="site-nav" aria-label="主导航">
            <a className="is-active" href="#products">精选产品</a>
            <a href="#service">服务保障</a>
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
                登录
              </Button>
            )}
            <Button type="primary" icon={<CustomerServiceOutlined />} onClick={consult}>
              咨询顾问
            </Button>
          </div>
        </div>
      </header>

      <main>
        <section className="hero-editorial">
          <div className="hero-editorial__inner">
            <div className="hero-copy hero-animate">
              <h1>
                选对保障，
                <br />
                从容面对生活
              </h1>
              <p>
                汇集优质保险产品，按家庭需求清晰比对条款与保障重点，保费以实际在线试算为准。
              </p>

              <div className="hero-cta-group">
                <Button
                  type="primary"
                  size="large"
                  icon={<CustomerServiceOutlined />}
                  onClick={consult}
                >
                  让顾问帮我选
                </Button>
                <a className="hero-secondary-link" href="#products">
                  浏览产品 <ArrowRightOutlined />
                </a>
              </div>
            </div>

            <aside className="hero-showcase hero-animate hero-animate--delay" aria-label="保障配置示例">
              <div className="hero-showcase__head">
                <div>
                  <span>配置示例参考</span>
                  <strong>30 岁有医保 · 均衡保障</strong>
                </div>
              </div>

              <div className="hero-showcase__rows">
                <div className="showcase-row">
                  <div>
                    <strong>百万医疗险</strong>
                    <small>住院与特药医疗报销</small>
                  </div>
                  <div className="showcase-row__right">
                    <span>¥290–¥360/年</span>
                    <div className="showcase-bar"><i style={{ width: '44%' }} /></div>
                  </div>
                </div>
                <div className="showcase-row">
                  <div>
                    <strong>重疾险</strong>
                    <small>重大疾病确诊一次性给付</small>
                  </div>
                  <div className="showcase-row__right">
                    <span>¥2,500–¥2,900/年</span>
                    <div className="showcase-bar"><i style={{ width: '84%' }} /></div>
                  </div>
                </div>
                <div className="showcase-row">
                  <div>
                    <strong>综合意外险</strong>
                    <small>意外伤残及门诊住院赔付</small>
                  </div>
                  <div className="showcase-row__right">
                    <span>¥180–¥296/年</span>
                    <div className="showcase-bar"><i style={{ width: '28%' }} /></div>
                  </div>
                </div>
              </div>

              <button type="button" className="hero-showcase__action" onClick={consult}>
                <span>为我定制方案</span>
                <ArrowRightOutlined />
              </button>
            </aside>
          </div>
        </section>

        <section className="products-section" id="products">
          <div className="products-heading">
            <div>
              <h2>精选保险产品</h2>
              <p>根据保障类型筛选或搜索合适产品</p>
            </div>
            <span className="products-count">
              {loading
                ? '正在加载...'
                : searchKeyword.trim()
                  ? `匹配 ${displayedProducts.length} / ${products.length} 款`
                  : `共 ${displayedProducts.length} 款产品`}
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
                placeholder="搜索产品、保司或亮点"
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
                  { label: '保费从低到高', value: 'price_asc' },
                  { label: '保费从高到低', value: 'price_desc' },
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
                  ? `未找到与“${searchKeyword.trim()}”相关的产品`
                  : '该类型暂无可展示的产品'
              }
            >
              {searchKeyword.trim() ? (
                <Button onClick={() => setSearchKeyword('')}>清空搜索</Button>
              ) : null}
            </Empty>
          )}
        </section>

        <section className="service-strip" id="service">
          <article className="trust-card">
            <strong>信息清晰</strong>
            <p>条款重点与保障责任集中展示，阅读更省心。</p>
          </article>
          <article className="trust-card">
            <strong>保费透明</strong>
            <p>按投保参数实时试算年缴费用，明明白白对比。</p>
          </article>
          <article className="trust-card">
            <strong>顾问协助</strong>
            <p>结合家庭成员与预算需求，梳理合适配置方案。</p>
          </article>
        </section>
      </main>

      <footer className="site-footer">
        保险产品信息仅供参考，具体保障责任以保险合同和实际投保页面为准。
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
                  保费试算
                </Button>,
                <Button
                  key="consult"
                  type="primary"
                  icon={<CustomerServiceOutlined />}
                  onClick={consult}
                >
                  咨询顾问
                </Button>,
              ]
            : null
        }
        onCancel={() => setSelectedProduct(null)}
        width={640}
      >
        {selectedProduct ? (
          <div className="product-detail">
            <div className="product-detail__banner">
              <div className="product-detail__banner-copy">
                <span className="detail-category-badge">
                  {CATEGORY_LABELS[selectedProduct.category]}
                </span>
                <h3>{selectedProduct.name}</h3>
              </div>
              <div className="product-detail__price-box">
                <span>年缴参考</span>
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
            </dl>

            <h4>产品亮点</h4>
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
