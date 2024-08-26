import React from 'react';
import { List, Typography } from 'antd';
import Link from 'next/link';
import Layout from '../../components/Layout';

const { Title } = Typography;

export default function ArticleList({ posts }) {
  return (
    <Layout>
      <Title>Articles</Title>
      <List
        itemLayout="horizontal"
        dataSource={posts}
        renderItem={(post) => (
          <List.Item>
            <List.Item.Meta
              title={<Link href={`/blog/${post.slug}`}>{post.title}</Link>}
              description={post.excerpt}
            />
          </List.Item>
        )}
      />
    </Layout>
  );
}

export async function getStaticProps() {
  // In a real application, you would fetch this data from your API or file system
  const posts = [
    { slug: 'about', title: 'About', excerpt: 'Learn about our approach...' },
    { slug: 'methodology-data', title: 'Data Preparation Methodology', excerpt: 'Learn about our approach...' },
    // { slug: 'methodology-training', title: 'Training Methodology', excerpt: 'Learn about our approach...' },
    // { slug: 'methodology-analysis', title: 'Analysis Methodology', excerpt: 'Learn about our approach...' },
  ];

  return {
    props: {
      posts,
    },
  };
}