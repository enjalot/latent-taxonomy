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
      {process.env.NODE_ENV === 'production' && (
            <>
              <script async src="https://www.googletagmanager.com/gtag/js?id=G-42X5GRN7RD"></script>
              <script
                dangerouslySetInnerHTML={{
                  __html: `
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){dataLayer.push(arguments);}
                    gtag('js', new Date());

                    gtag('config', 'G-42X5GRN7RD');
                  `,
                }}
              />
            </>
          )}
    </Head>
    <Component {...pageProps} />
  </>)
}

export default MyApp;