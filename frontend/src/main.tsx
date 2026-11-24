import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConfigProvider } from 'antd';
import 'antd/dist/reset.css';
import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#0F3D38',
          colorInfo: '#0F3D38',
          colorWarning: '#B88940',
          colorSuccess: '#146C5B',
          colorError: '#B44726',
          colorBgLayout: '#F6F3EC',
          colorBgContainer: '#FFFDF9',
          colorBgElevated: '#FFFDF9',
          colorText: '#142422',
          colorTextSecondary: '#5A6763',
          colorBorder: '#E5DEC9',
          borderRadius: 12,
          fontFamily: '"Inter", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
        },
        components: {
          Button: {
            borderRadius: 10,
            controlHeight: 40,
            fontWeight: 600,
          },
          Segmented: {
            trackBg: '#ECE6D8',
            itemSelectedBg: '#0F3D38',
            itemSelectedColor: '#FAF6EE',
            itemColor: '#56635E',
            borderRadius: 10,
          },
          Modal: {
            borderRadiusLG: 18,
            contentBg: '#FFFDF9',
            headerBg: '#FFFDF9',
          },
          Drawer: {
            colorBgElevated: '#FFFDF9',
          },
          Input: {
            borderRadius: 10,
            colorBgContainer: '#FFFDF9',
            activeBorderColor: '#0F3D38',
            hoverBorderColor: '#B88940',
          },
          InputNumber: {
            borderRadius: 10,
            colorBgContainer: '#FFFDF9',
            activeBorderColor: '#0F3D38',
            hoverBorderColor: '#B88940',
          },
          Select: {
            borderRadius: 10,
            colorBgContainer: '#FFFDF9',
            optionSelectedBg: '#E9F2EF',
          },
          Steps: {
            colorPrimary: '#0F3D38',
          },
          Tabs: {
            inkBarColor: '#B88940',
            itemSelectedColor: '#0F3D38',
            itemActiveColor: '#0F3D38',
            itemHoverColor: '#0F3D38',
          },
          Descriptions: {
            labelBg: '#F4EFE4',
            contentColor: '#142422',
            labelColor: '#5A6763',
          },
          Alert: {
            borderRadiusLG: 12,
          },
          Tag: {
            borderRadiusSM: 8,
          },
          Card: {
            borderRadiusLG: 16,
            colorBgContainer: '#FFFDF9',
          },
        },
      }}
    >
      <App />
    </ConfigProvider>
  </React.StrictMode>,
);
