import React, { useState, useEffect, useMemo } from 'react';
import { AutoComplete, Button } from 'antd';
import { CloseCircleOutlined } from '@ant-design/icons';
import { interpolateTurbo } from 'd3-scale-chromatic';

import styles from './FeatureSelect.module.css';

const FeatureSelect = ({ 
  options, 
  value, 
  onSelect = () => {},
  onFilter = () => {}
} = {}) => {
  const [searchValue, setSearchValue] = useState('');
  const [inputValue, setInputValue] = useState('');

  const handleSearch = (value) => {
    setSearchValue(value);
    setInputValue(value);
  };

  const handleSelect = (value) => {
    const selectedOption = options.find(option => option.feature === +value);
    onSelect(selectedOption);
    onFilter([])
    setInputValue(selectedOption.label);
  };
  const handleClear = () => {
    onSelect(null);
    setInputValue('');
  };

  const filteredOptions = useMemo(() => options.filter(option =>
    option.label.toLowerCase().includes(searchValue.toLowerCase()) ||
    option.feature.toString().includes(searchValue)
  ), [options, searchValue]);

  useEffect(() => {
    if(filteredOptions.length !== options.length) {
      onFilter(filteredOptions);
    } else {
      onFilter([]);
    }
  }, [filteredOptions ]);

  useEffect(() => {
    if(value) {
      setInputValue(value.label);
    } else {
      setInputValue('');
    }
  }, [value]);

  return (
    <div className={styles.container}>
      <AutoComplete
        options={filteredOptions.map(option => ({
          value: String(option.feature),
          label: (
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor: interpolateTurbo(option.order),
                  marginRight: '8px',
                }}
              />
              <span style={{ width: 'calc(100% - 20px)' }}>{option.feature}: {option.label}</span>
            </div>
          ),
        }))}
        value={inputValue}
        onSearch={handleSearch}
        onSelect={handleSelect}
        placeholder="Select a feature"
        style={{ width: '100%' }}
      />
      <Button
        onClick={handleClear}
        disabled={!inputValue}
        icon={<CloseCircleOutlined />}
        className={styles.clearButton}
      />
    </div>
  );
};

export default FeatureSelect;
