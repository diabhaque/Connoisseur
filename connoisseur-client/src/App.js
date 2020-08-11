import React, { useState } from 'react';
import AppRouter from './routers/AppRouter'
import ImageContext from './context/image.context'

function App() {
  const [image, setImage]=useState({
    image: undefined
  })

  return (
    <ImageContext.Provider 
      value={[image, setImage]}>
      <AppRouter/>
    </ImageContext.Provider>
  );
}

export default App;