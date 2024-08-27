import React, { useEffect, useState, useMemo} from 'react';
import { Typography, Descriptions } from 'antd';
const { asyncBufferFromUrl, parquetRead } = await import('hyparquet')

import styles from './FeatureDetails.module.css';

const { Title, Paragraph } = Typography;

const FeatureDetails = ({ feature }) => {
  const [samples, setSamples] = useState([]);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const buffer = await asyncBufferFromUrl("/models/NOMIC_FWEDU_25k/features.parquet");
        const data = await parquetRead({
          file: buffer,
          onComplete: data => {
            const fts = data.map(f => ({
              feature: parseInt(f[0]),
              max_activation: f[1],
              x: f[2],
              y: f[3],
              top10_x: f[4],
              top10_y: f[5],
              label: f[6],
              order: f[7],
            }));
            setSamples(fts.slice(0, 10));
            // setFeatures(fts);
          }
        });
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, [feature]);

  useEffect(() => {
    console.log("samples", samples.length)
  }, [samples])

  return (
    <div className={styles.details}>
      {!feature ? <Paragraph>Select a feature to view details.</Paragraph>: <>
        <Title level={4}>{feature.feature}: {feature.label}</Title>
        { samples.length && <div className={styles.samples}>
          <Descriptions title="Samples" column={1}>
            {samples.map(sample => (
              <Descriptions.Item label={sample.feature}>
                <div>
                  <div>Max Activation: {sample.max_activation}</div>
                  <div>Top 10 Activation: {sample.top10_x}, {sample.top10_y}</div>
                  <div>Activation: {sample.x}, {sample.y}</div>
                </div>
              </Descriptions.Item>
            ))}
          </Descriptions>
        </div>}
      </>}
    </div>
  );
};

export default FeatureDetails;
