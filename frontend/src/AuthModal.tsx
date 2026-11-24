import { LockOutlined, MailOutlined, SafetyCertificateOutlined, UserOutlined } from '@ant-design/icons';
import { Alert, Form, Input, Modal, Tabs } from 'antd';
import { useState } from 'react';

import { loginUser, registerUser } from './api';
import { toAuthSession } from './auth';
import type { AuthSession } from './types';

interface AuthModalProps {
  open: boolean;
  onCancel: () => void;
  onSuccess: (session: AuthSession) => void;
}

interface LoginFormValues {
  username: string;
  password: string;
}

interface RegisterFormValues {
  username: string;
  email: string;
  password: string;
}

export function AuthModal({
  open,
  onCancel,
  onSuccess,
}: AuthModalProps) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleLogin(values: LoginFormValues) {
    setSubmitting(true);
    setError(null);
    try {
      const payload = await loginUser(values.username.trim(), values.password);
      onSuccess(toAuthSession(payload));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '登录失败，请检查账号和密码');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRegister(values: RegisterFormValues) {
    setSubmitting(true);
    setError(null);
    try {
      const payload = await registerUser(
        values.username.trim(),
        values.email.trim(),
        values.password,
      );
      onSuccess(toAuthSession(payload));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '注册失败，请检查输入信息');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Modal
      open={open}
      title={null}
      footer={null}
      width={420}
      onCancel={onCancel}
      destroyOnHidden
      centered
    >
      <div className="auth-modal__header">
        <div className="auth-modal__emblem">
          <SafetyCertificateOutlined />
        </div>
        <h2>欢迎来到安心保</h2>
        <p>登录后即可使用智能顾问与保单服务（演示账号：demo / 123456）</p>
      </div>

      <Tabs
        centered
        activeKey={mode}
        onChange={(key) => {
          setMode(key as 'login' | 'register');
          setError(null);
        }}
        items={[
          { key: 'login', label: '账号登录' },
          { key: 'register', label: '注册新账号' },
        ]}
      />

      {error ? (
        <Alert className="auth-modal__alert" showIcon type="error" message={error} />
      ) : null}

      {mode === 'login' ? (
        <Form<LoginFormValues>
          layout="vertical"
          initialValues={{ username: 'demo', password: '123456' }}
          onFinish={(vals) => void handleLogin(vals)}
          requiredMark={false}
        >
          <Form.Item
            label="用户名或邮箱"
            name="username"
            rules={[{ required: true, message: '请输入用户名或邮箱' }]}
          >
            <Input prefix={<UserOutlined />} size="large" placeholder="demo 或邮箱" />
          </Form.Item>
          <Form.Item
            label="密码"
            name="password"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password prefix={<LockOutlined />} size="large" placeholder="至少 6 位密码" />
          </Form.Item>
          <button className="auth-submit" type="submit" disabled={submitting}>
            {submitting ? '登录中...' : '登录'}
          </button>
        </Form>
      ) : (
        <Form<RegisterFormValues>
          layout="vertical"
          onFinish={(vals) => void handleRegister(vals)}
          requiredMark={false}
        >
          <Form.Item
            label="用户名"
            name="username"
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input prefix={<UserOutlined />} size="large" placeholder="请输入用户名" />
          </Form.Item>
          <Form.Item
            label="邮箱"
            name="email"
            rules={[
              { required: true, message: '请输入邮箱' },
              { type: 'email', message: '请输入有效邮箱地址' },
            ]}
          >
            <Input prefix={<MailOutlined />} size="large" placeholder="name@example.com" />
          </Form.Item>
          <Form.Item
            label="密码"
            name="password"
            rules={[{ required: true, min: 6, message: '密码至少需要 6 位' }]}
          >
            <Input.Password prefix={<LockOutlined />} size="large" placeholder="设置至少 6 位密码" />
          </Form.Item>
          <button className="auth-submit" type="submit" disabled={submitting}>
            {submitting ? '注册中...' : '注册并登录'}
          </button>
        </Form>
      )}
    </Modal>
  );
}
