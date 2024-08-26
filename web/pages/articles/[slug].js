import React from 'react';
import { Typography } from 'antd';
import { getMDXComponent } from 'mdx-bundler/client';
import Layout from '../../components/Layout';
import { getPostBySlug, getAllPostSlugs } from '../../lib/posts';

const { Title } = Typography;

export default function BlogPost({ code, frontmatter }) {
  const Component = React.useMemo(() => getMDXComponent(code), [code]);

  return (
    <Layout>
      <Title>{frontmatter.title}</Title>
      <Component />
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