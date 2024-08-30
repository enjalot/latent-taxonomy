import React, { useEffect } from 'react';
import { useRouter } from 'next/router';
import { Typography, Row, Col } from 'antd';
import Image from 'next/image';
import dynamic from 'next/dynamic';
import { getMDXComponent } from 'mdx-bundler/client';
import Layout from '../../components/Layout';
import { getPostBySlug, getAllPostSlugs } from '../../lib/posts';

const { Title } = Typography;

import styles from './article.module.css';

// Custom Image component that uses the basePath
const CustomImage = (props) => {
  const router = useRouter();
  const src = `${router.basePath}${props.src}`;
  return <Image {...props} src={src} />;
};

// Custom MDX component
// Crazy stuff to get the mdx importing stuff
const MDXComponent = ({ code, components }) => {
  const [Component, setComponent] = React.useState(() => () => null);

  React.useEffect(() => {
    // Polyfill for process.env in the browser
    if (typeof window !== 'undefined' && !window.process) {
      window.process = { env: {} };
    }
    // Only import and render on the client-side
    import('mdx-bundler/client').then(({ getMDXComponent }) => {
      setComponent(() => getMDXComponent(code));
    });
  }, [code]);

  return <Component components={components} />;
};

export default function BlogPost({ code, frontmatter }) {
  const components = {
    img: CustomImage,
    h1: (props) => <h1 id={props.children.replace(/\s+/g, '-').toLowerCase().replace(/[?]/g, '')} {...props} />,
    h2: (props) => <h2 id={props.children.replace(/\s+/g, '-').toLowerCase().replace(/[?]/g, '')} {...props} />,
    h3: (props) => <h3 id={props.children.replace(/\s+/g, '-').toLowerCase().replace(/[?]/g, '')} {...props} />,
  };

  useEffect(() => {
    // Implement smooth scrolling for hash links with offset
    const handleHashChange = () => {
      const hash = window.location.hash;
      if (hash) {
        const element = document.getElementById(hash.slice(1));
        if (element) {
          const headerOffset = 80; // Adjust this value based on your header height
          const elementPosition = element.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

          window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
          });
        }
      }
    };

    // Call the function on initial load and when the hash changes
    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);

    return () => {
      window.removeEventListener('hashchange', handleHashChange);
    };
  }, []);

  return (
    <Layout>
      <Row justify="center">
        <Col xs={24} sm={22} md={20} lg={18} xl={16}>
          <article className={styles.articleCard}>
            <Title>{frontmatter.title}</Title>
            {/* <Component components={components} /> */}
            <MDXComponent code={code} components={components} />
          </article>
        </Col>
      </Row>
    </Layout>
  );
}

export async function getStaticPaths() {
  const paths = getAllPostSlugs();
  return {
    paths,
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const postData = await getPostBySlug(params.slug);
  return {
    props: {
      ...postData,
    },
  };
}