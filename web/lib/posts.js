import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const postsDirectory = path.join(process.cwd(), 'posts');

export function getAllPostSlugs() {
  const fileNames = fs.readdirSync(postsDirectory);
  return fileNames.map(fileName => {
    return {
      params: {
        slug: fileName.replace(/\.mdx$/, '')
      }
    };
  });
}

export function getPostSlugs() {
  const fileNames = fs.readdirSync(postsDirectory);
  return fileNames.map(fileName => fileName.replace(/\.mdx$/, ''));
}

export async function getPostBySlug(slug) {
  const fullPath = path.join(postsDirectory, `${slug}.mdx`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');

  const { data: frontmatter, content } = matter(fileContents);

  const { bundleMDX } = await import('mdx-bundler');
  const { code } = await bundleMDX({
    source: content,
    cwd: postsDirectory,
  });

  return {
    slug,
    frontmatter,
    code
  };
}

export async function getAllPosts() {
  const slugs = getPostSlugs();
  const posts = await Promise.all(slugs.map(async (slug) => getPostBySlug(slug)));

  // Sort posts by date
  return posts.sort((post1, post2) => {
    if (post1.frontmatter.date > post2.frontmatter.date) {
      return -1;
    } else {
      return 1;
    }
  });
}