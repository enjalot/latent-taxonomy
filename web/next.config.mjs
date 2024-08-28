import withMDX from '@next/mdx';
import withTM from 'next-transpile-modules';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',
  pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdx'],
  transpilePackages: ['antd', '@ant-design', 'rc-util', 'rc-pagination', 'rc-picker'],
  assetPrefix: process.env.NODE_ENV === 'production' ? '/latent-taxonomy/' : '',
  basePath: process.env.NODE_ENV === 'production' ? '/latent-taxonomy' : '',
  webpack: (config) => {
    config.resolve.fallback = { fs: false, path: false };
    return config;
  },
  // babel: {
  //   presets: ['next/babel'],
  //   plugins: [],
  // },
}

export default withTM(withMDX())(nextConfig);