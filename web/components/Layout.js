import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Layout as AntLayout, Menu } from 'antd';
import Link from 'next/link';

import styles from './Layout.module.css';

const { Header, Content, Footer } = AntLayout;

const MainLayout = ({ children }) => {
  const router = useRouter();
  const getInitialSelectedKey = (pathname) => {
    switch (pathname) {
      case '/':
        return '1';
      case '/articles/[slug]':
        return router.query?.slug == "about" ? '2' : '2';
      case '/articles':
        return '3';
      default:
        return '1';
    }
  };

  const menuItems = [
    { key: '1', label: <Link href="/">Latent Taxonomy</Link> },
    { key: '2', label: <Link href="/articles/about">About</Link> },
    // { key: '3', label: <Link href="/articles">Methodology</Link> },
  ];

  const [selectedKey, setSelectedKey] = useState(getInitialSelectedKey(router.pathname));

  useEffect(() => {
    setSelectedKey(getInitialSelectedKey(router.pathname));
  }, [router.pathname]);

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