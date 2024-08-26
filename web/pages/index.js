import React, { useState, useRef, useEffect } from 'react';
import { Typography, Card, Row, Col, Select } from 'antd';
import Layout from '../components/Layout';
import VirtualizedSelect from '../components/VirtualizedSelect';
import FeatureSelect from '../components/FeatureSelect';

const { asyncBufferFromUrl, parquetRead } = await import('hyparquet')


import styles from './index.module.css';

const { Title } = Typography;

const models = [
   { label: "NOMIC_FWEDU_25k", value: "NOMIC_FWEDU_25k" },
   //{ label: "NOMIC_FWEDU_100k", value: "NOMIC_FWEDU_100k" },
]

// This is a placeholder for your actual visualization component
const VisualizationPlaceholder = ({ width, height }) => (
  <div 
    style={{ 
      width: width, 
      height: height, 
      backgroundColor: '#f0f0f0', 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center',
      border: '1px dashed #ccc'
    }}
  >
    <p>Your Visualization Here</p>
    <p>Width: {width}px, Height: {height}px</p>
  </div>
);


export default function Home() {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const mainCardRef = useRef(null);

  useEffect(() => {
    const updateDimensions = () => {
      if (mainCardRef.current) {
        const { offsetWidth, offsetHeight } = mainCardRef.current;
        console.log("DIMENISONS", mainCardRef.current.clientHeight, mainCardRef.current.offsetHeight)
        setDimensions({
          width: offsetWidth,
          height: offsetHeight - 57, // Subtracting the Card header height
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);

    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const [features, setFeatures] = useState([])
  useEffect(() => {
    const asyncRead = async () => {
      const buffer = await asyncBufferFromUrl("/models/NOMIC_FWEDU_25k/features.parquet")
      const data = await parquetRead({
        file: buffer,
        onComplete: data => {
          setFeatures(data.map(f => {
            return {
              feature: parseInt(f[0]),
              max_activation: f[1],
              x: f[2],
              y: f[3],
              label: f[4],
              order: parseInt(f[5])
            }
          }).sort((a,b) => a.order - b.order))
        }
      })
    }
    asyncRead()
  }, [])
  useEffect(() => {
    console.log("FEATURES", features)
  }, [features])


  const [selectedFeature, setSelectedFeature] = useState(features[0]);
  const handleFeatureSelect = (feature) => {
    setSelectedFeature(feature);
    console.log("FEATURE SELECTED", feature)
  }

  const [selectedModel, setSelectedModel] = useState(models[0])
  const handleModelSelect = (model) => {
    setSelectedModel(model)
  }
  const [modelMetadata, setModelMetadata] = useState(null)
  useEffect(() => {
    const asyncRead = async () => {
      const meta = await fetch(`/models/${selectedModel.label}/metadata.json`).then(r => r.json())
      setModelMetadata(meta)
    }
    asyncRead()
  }, [selectedModel])

  return (
    <Layout>
      <div className={styles.homeContainer}>
        <Title className={styles.pageTitle}>Latent Taxonomy</Title>
        <Row className={styles.fullHeightRow} gutter={[24, 24]}>
          <Col xs={24} lg={12} className={styles.fullHeightCol} >
            <Card title={
              <div className={styles.modelTitle}>
                <Select options={models} value={models[0]} onChange={handleModelSelect} />
                {modelMetadata && (
                  <div className={styles.modelMetadata}>
                    <span>{modelMetadata.num_latents} ({modelMetadata.expansion}x)</span>
                    <span>{modelMetadata.source_model} ({modelMetadata.d_in})</span>
                  </div>
                )}
              </div>
              } 
              className={styles.fullHeightCard} 
              ref={mainCardRef}>
              <VisualizationPlaceholder width={dimensions.width} height={dimensions.height} />
            </Card>
          </Col>
          <Col xs={24} lg={12} className={styles.fullHeightCol}>
          <Card 
              title={
                <FeatureSelect 
                  options={features} 
                  value={selectedFeature} 
                  onSelect={handleFeatureSelect} 
                />
              } 
              className={styles.fullHeightCard}
            >
              <div className={styles.scrollableContent}>
                {/* <VirtualizedSelect options={options} value={options[0]} onSelect={handleFeatureSelect} /> */}
                {/* <Select options={options} value={options[0]} onSelect={handleFeatureSelect} /> */}
                {/* <FeatureSelect options={options} value={selectedFeature} onSelect={handleFeatureSelect} /> */}
                {/* Add more content here to demonstrate scrolling */}
                {[...Array(100)].map((_, i) => (
                  <p key={i}>Scrollable content line {i + 1}</p>
                ))}
              </div>
            </Card>
          </Col>
        </Row>
      </div>
    </Layout>
  );
}