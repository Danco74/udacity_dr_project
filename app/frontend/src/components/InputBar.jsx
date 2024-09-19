import React, { useState } from 'react';

const InputBar = ({ onInputChange }) => {
  const [inputValue, setInputValue] = useState('');  // State for input value
  const [loading, setLoading] = useState(false);  // State for loading animation

  // Handle input change
  const handleChange = (event) => setInputValue(event.target.value);

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();  // Prevent page reload on form submission
    if (inputValue.trim() !== '') {
      setLoading(true);  // Start loading animation
      try {
        await onInputChange(inputValue);  // Call parent function to handle input value
      } finally {
        setLoading(false);  // Stop loading animation after the request
      }
    }
  };

  return (
    <div className="flex flex-col items-center max-w-md mx-auto mt-5">
      <form onSubmit={handleSubmit} className="flex w-full">
        <div className="flex items-center w-full">
          <div className="flex flex-grow px-4 py-3 border-2 shadow-md border-gray-300 overflow-hidden">
            <input
              type="text"
              onChange={handleChange}
              value={inputValue}
              placeholder="Classify..."
              className="w-full outline-none bg-transparent text-gray-600 text-sm ml-3"
            />
          </div>
          <button
            type="submit"
            className="flex ml-2 px-4 py-3 bg-gray-600 text-white rounded-md shadow-md hover:bg-blue-600"
          >
            {/* Icon with conditional spinning animation when loading */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className={`size-6 mr-2 text-gray-300 ${loading ? 'animate-spin' : ''}`}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23-.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
              />
            </svg>
            Classify
          </button>
        </div>
      </form>

      {/* Example list below the input field */}
      <div className="mt-4 w-full">
        <p className="text-gray-500 text-sm">Examples:</p>
        <ul className="list-disc list-inside text-gray-500 text-sm mt-2">
          <li>"Send help to the earthquake victims."</li>
          <li>"We need food and water in the affected area."</li>
          <li>"There is a fire in the building, we need assistance!"</li>
          <li>"Medical supplies are urgently required."</li>
        </ul>
      </div>
    </div>
  );
};

export default InputBar;
