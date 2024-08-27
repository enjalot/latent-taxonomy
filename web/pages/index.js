import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Typography, Card, Row, Col, Select } from 'antd';
import { interpolateTurbo } from 'd3-scale-chromatic';
import {Tooltip} from 'react-tooltip';

import Layout from '../components/Layout';
import FeatureSelect from '../components/FeatureSelect';
import Scatter from '../components/Scatter';
import StaticScatter from '../components/StaticScatter';

const { asyncBufferFromUrl, parquetRead } = await import('hyparquet')


import styles from './index.module.css';

const { Title } = Typography;

// unfortunately regl-scatter doesn't even render in iOS
const isIOS = () => {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}
// let's warn mobile users (on demo in read-only) that desktop is better experience
const isMobileDevice = () => {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

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
        // console.log("DIMENISONS", mainCardRef.current.clientHeight, mainCardRef.current.offsetHeight)
        setDimensions({
          width: offsetWidth,
          height: offsetHeight, // Subtracting the Card header height
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);

    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const [features, setFeatures] = useState([])
  const [points, setPoints] = useState([])
  useEffect(() => {
    const asyncRead = async () => {
      const buffer = await asyncBufferFromUrl("/models/NOMIC_FWEDU_25k/features.parquet")
      const data = await parquetRead({
        file: buffer,
        onComplete: data => {
          // let pts = []
          let fts = data.map(f => {
            // pts.push([f[2], f[3], parseInt(f[5])])
            return {
              feature: parseInt(f[0]),
              max_activation: f[1],
              x: f[2],
              y: f[3],
              label: f[4],
              order: parseInt(f[5])
            }
          })
          // .sort((a,b) => a.order - b.order)
          let pts = fts.map(f => [f.x, f.y, f.x])
          console.log("POOINTS", pts)
          setFeatures(fts)
          setPoints(pts)
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

  // ====================================================================================================
  // Scatterplot related logic
  // ====================================================================================================
  // this is a reference to the regl scatterplot instance
  // so we can do stuff like clear selections without re-rendering
  const [scatter, setScatter] = useState({})
  const [xDomain, setXDomain] = useState([-1, 1]);
  const [yDomain, setYDomain] = useState([-1, 1]);
  const handleView = useCallback((xDomain, yDomain) => {
    setXDomain(xDomain);
    setYDomain(yDomain);
  }, [setXDomain, setYDomain])
  // Selection via Scatterplot
  // indices of items selected by the scatter plot
  const [selectedIndices, setSelectedIndices] = useState([]);

  const handleSelected = useCallback((indices) => {
    // console.log("handle selected", indices)
    setSelectedIndices(indices)
  }, [setSelectedIndices])

  // Hover via scatterplot or tables
  // index of item being hovered over
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [hoveredFeature, setHoveredFeature] = useState(null);
  // const [tooltipContent, setTooltipContent] = useState('');

  useEffect(() => {
    if (hoveredIndex !== null && hoveredIndex !== undefined) {
      setHovered(hoveredIndex);
      const feature = features[hoveredIndex];
      console.log("hovered", hoveredIndex, feature)
      if (feature) {
        const xPos = ((feature.x - xDomain[0]) / (xDomain[1] - xDomain[0])) * dimensions.width;
        const yPos = ((feature.y - yDomain[1]) / (yDomain[0] - yDomain[1])) * (dimensions.height) + 57;
        setTooltipPosition({ x: xPos - .5*16, y: yPos - .67*16 });
        setHoveredFeature(feature)
      }
    } else {
      setHovered(null);
      setHoveredFeature(null)
    }
  }, [hoveredIndex, setHovered, features, xDomain, yDomain, dimensions]);

  const handleHover = useCallback((index) => {
    setHoveredIndex(index);
  }, [setHoveredIndex])



  return (
    <Layout>
      <div className={styles.homeContainer}>
        <Title className={styles.pageTitle}>Latent Taxonomy</Title>
        <Row className={styles.fullHeightRow} gutter={[24, 24]}>
          <Col xs={24} lg={12} className={styles.fullHeightCol} >
            <Card title={
              <div className={styles.modelTitle}>
                <Select options={models} value={models[0]} onChange={handleModelSelect} data-tooltip-id="modelTooltip" />
                {modelMetadata && (
                  <div className={styles.modelMetadata}>
                    <span>{modelMetadata.num_latents} ({modelMetadata.expansion}x)</span>
                    <span>{modelMetadata.source_model} ({modelMetadata.d_in})</span>
                  </div>
                )}
                <Tooltip id="modelTooltip">
                  Select a model
                </Tooltip>
              </div>
              } 
            className={styles.fullHeightCard} 
            ref={mainCardRef}>
              <div className="scatters" style={{ width: dimensions.width, height: dimensions.height }}>
                {points.length ? <>
                  { !isIOS() ? <Scatter
                    points={points}
                    duration={2000}
                    width={dimensions.width}
                    height={dimensions.height}
                    colorScaleType="continuous"
                    // colorScaleType="categorical"
                    colorInterpolator={interpolateTurbo}
                    pointScale={2}
                    onScatter={setScatter}
                    onView={handleView}
                    onSelect={handleSelected}
                    onHover={handleHover}
                  /> : <StaticScatter
                    points={points}
                    fill="gray"
                    size="8"
                    xDomain={xDomain}
                    yDomain={yDomain}
                  width={dimensions.width}
                  height={dimensions.height}
                /> }
                </> : null }
            </div>
            
            <div
              data-tooltip-id="featureTooltip"
              style={{
                position: 'absolute',
                left: tooltipPosition.x,
                top: tooltipPosition.y,
                pointerEvents: 'none',
              }}
            >⭐️</div>
            <Tooltip id="featureTooltip" 
              isOpen={hoveredIndex !== null}
              style={{
                position: 'absolute',
                left: tooltipPosition.x,
                top: tooltipPosition.y,
                pointerEvents: 'none',
              }}
            >
              {hoveredFeature?.feature}: {hoveredFeature?.label}
            </Tooltip>
            
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
                {/* {[...Array(100)].map((_, i) => (
                  <p key={i}>Scrollable content line {i + 1}</p>
                ))} */}
              </div>
            </Card>
          </Col>
        </Row>

      </div>
    </Layout>
  );
}