import React from 'react'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Layout from './components/Layout'
import Classification from './pages/Classification'
import DbInfo from './pages/DbInfo'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { path: '/', element: <Classification /> },
      { path: '/dbinfo', element: <DbInfo /> }
    ],
  },
])

const App = () => {
  return (
    <RouterProvider router={router} />
  )
}

export default App