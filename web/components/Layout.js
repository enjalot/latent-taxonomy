import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Layout as AntLayout, Menu } from 'antd';
import Link from 'next/link';

import styles from './Layout.module.css';

const { Header, Content, Footer } = AntLayout;

const MainLayout = ({ children }) => {
  const router = useRouter();

  const menuItems = [
    { key: '/', label: <Link href="/">Latent Interfaces</Link> },
    { key: '/articles/about', label: <Link href="/articles/about">About</Link> },
    { key: '/articles/sae-intuition', label: <Link href="/articles/sae-intuition">Intro to SAE</Link> },
  ];
  
  const [selectedKey, setSelectedKey] = useState(router.asPath);

  useEffect(() => {
    setSelectedKey(router.asPath);
  }, [router.asPath]);

  return (
    <AntLayout className={styles.layout}>
    <Header className={styles.header}>
      <div className={styles.logo} />
        <Menu theme="dark" mode="horizontal" selectedKeys={[selectedKey]} items={menuItems} />
    </Header>
    <Content className={styles.content}>
      <div className={styles.contentWrapper}>
        {children}
      </div>
    </Content>
    <Footer className={styles.footer}>
      ©2024 Latent Interfaces | @enjalot
    </Footer>
  </AntLayout>
)
}

export default MainLayout;