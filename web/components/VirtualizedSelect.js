import React, { useState, useMemo, useEffect } from 'react';
import { List, AutoSizer } from 'react-virtualized';
import { ChevronDown, Search, X } from 'lucide-react';

const VirtualizedSelect = ({ options, onSelect, value }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [internalValue, setInternalValue] = useState(value);

  useEffect(() => {
    setInternalValue(value);
  }, [value]);

  const toggleSelect = () => setIsOpen(!isOpen);

  const handleSelect = (option) => {
    setInternalValue(option);
    onSelect(option);
    setIsOpen(false);
  };

  const handleClear = (e) => {
    e.stopPropagation();
    setInternalValue(null);
    onSelect(null);
  };

  const filteredOptions = useMemo(() => {
    return options.filter(option =>
      option.label.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [options, searchTerm]);

  const renderOption = ({ index, style }) => {
    const option = filteredOptions[index];
    return (
      <div
        key={option.id}
        style={style}
        className={`px-4 py-2 cursor-pointer hover:bg-blue-50 flex items-center ${
          internalValue && internalValue.id === option.id ? 'bg-blue-100' : ''
        }`}
        onClick={() => handleSelect(option)}
      >
        <span
          className="w-3 h-3 rounded-full mr-2"
          style={{ backgroundColor: option.color }}
        ></span>
        {option.label}
      </div>
    );
  };

  return (
    <div className="relative w-64">
      <div
        className="border rounded-md px-3 py-2 flex items-center cursor-pointer bg-white hover:border-blue-400 focus-within:border-blue-400 focus-within:ring-1 focus-within:ring-blue-400"
        onClick={toggleSelect}
      >
        <Search size={16} className="text-gray-400 mr-2" />
        <input
          type="text"
          className="flex-grow outline-none"
          placeholder={internalValue ? internalValue.label : "Select an option"}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onClick={(e) => e.stopPropagation()}
        />
        {internalValue && (
          <X
            size={16}
            className="text-gray-400 hover:text-gray-600 cursor-pointer"
            onClick={handleClear}
          />
        )}
        <ChevronDown size={16} className="text-gray-400 ml-2" />
      </div>
      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white border rounded-md shadow-lg max-h-60 overflow-hidden">
          <AutoSizer disableHeight>
            {({ width }) => (
              <List
                width={width}
                height={240}
                rowCount={filteredOptions.length}
                rowHeight={40}
                rowRenderer={renderOption}
              />
            )}
          </AutoSizer>
        </div>
      )}
    </div>
  );
};

export default VirtualizedSelect;