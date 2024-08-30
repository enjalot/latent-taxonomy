import 'antd/dist/reset.css';
import '../styles/globals.css';
import { useMemo } from 'react'; 
import { useRouter } from 'next/router';
import Head from 'next/head';

function MyApp({ Component, pageProps }) {
  const router = useRouter(); // Use useRouter from next/router
  const basePath = useMemo(() => router.basePath, [router])

  return (<>
    <Head>
      <link rel="icon" href={`${basePath}/favicon.ico`} />
    </Head>
    <Component {...pageProps} />
  </>)
}

export default MyApp;