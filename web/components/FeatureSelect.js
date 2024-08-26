import React, { useState, useEffect, useMemo } from 'react';
import { AutoComplete, Button } from 'antd';
import { CloseCircleOutlined } from '@ant-design/icons';
import { interpolateTurbo } from 'd3-scale-chromatic';

import styles from './FeatureSelect.module.css';

const FeatureSelect = ({ options, value, onSelect }) => {
  const [searchValue, setSearchValue] = useState('');

  const handleSearch = (value) => {
    setSearchValue(value);
  };

  const handleSelect = (value) => {
    const selectedOption = options.find(option => option.feature === value);
    onSelect(selectedOption);
  };
  const handleClear = () => {
    onSelect(null);
  };

  const filteredOptions = useMemo(() => options.filter(option =>
    option.label.toLowerCase().includes(searchValue.toLowerCase()) ||
    option.feature.toString().includes(searchValue)
  ), [options, searchValue]);

  useEffect(() => {
    console.log("current value", value)
  }, [value])

  return (
    <div className={styles.container}>
      <AutoComplete
        options={filteredOptions.map(option => ({
          value: option.feature,
          label: (
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor: interpolateTurbo(option.order / (filteredOptions.length - 1)),
                  marginRight: '8px',
                }}
              />
              <span style={{ width: 'calc(100% - 20px)' }}>{option.feature}: {option.label}</span>
            </div>
          ),
        }))}
        value={value?.feature}
        onSearch={handleSearch}
        onSelect={handleSelect}
        placeholder="Select a feature"
        style={{ width: '100%' }}
      />
      <Button
        onClick={handleClear}
        disabled={!value}
        icon={<CloseCircleOutlined />}
        className={styles.clearButton}
      />
    </div>
  );
};

export default FeatureSelect;
