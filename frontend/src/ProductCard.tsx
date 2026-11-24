import { CheckCircleOutlined, RightOutlined, TeamOutlined } from '@ant-design/icons';
import { useState } from 'react';

import { CATEGORY_LABELS } from './categoryLabels';
import type { Product } from './types';

interface ProductCardProps {
  product: Product;
  onView: (product: Product) => void;
}

export function ProductCard({ product, onView }: ProductCardProps) {
  const [imageFailed, setImageFailed] = useState(false);

  return (
    <article className="product-card">
      <div className="product-card__media">
        {!imageFailed && product.image_url ? (
          <img
            src={product.image_url}
            alt=""
            loading="lazy"
            referrerPolicy="no-referrer"
            onError={() => setImageFailed(true)}
          />
        ) : (
          <div className="product-card__fallback">
            <span>{CATEGORY_LABELS[product.category]}</span>
          </div>
        )}
        <div className="product-card__scrim" />
        <strong className="product-card__badge">{CATEGORY_LABELS[product.category]}</strong>
        <span className="product-card__insurer-tag">{product.insurer}</span>
      </div>

      <div className="product-card__body">
        <h3 title={product.name}>{product.name}</h3>
        <p className="product-card__description">{product.description}</p>

        <div className="product-card__audience">
          <TeamOutlined />
          <span>适合：{product.target_group}</span>
        </div>

        <ul className="product-card__highlights">
          {product.highlights.slice(0, 3).map((highlight) => (
            <li key={highlight}>
              <CheckCircleOutlined />
              <span>{highlight}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="product-card__footer">
        <div className="product-card__price">
          <span>年缴保费参考</span>
          <div>
            <strong>¥{product.min_premium.toLocaleString('zh-CN')}</strong>
            <small>起</small>
          </div>
        </div>
        <button type="button" className="product-card__cta" onClick={() => onView(product)}>
          查看详情 <RightOutlined />
        </button>
      </div>
    </article>
  );
}
