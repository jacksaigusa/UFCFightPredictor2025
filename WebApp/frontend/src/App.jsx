import { useState, useEffect } from 'react'
import PredictionList from './PredictionList'
import PastPredictionList from './PastPredictionList'
import './App.css'

function App() {
  const [predictions, setPredictions] = useState([])
  const [pastPredictions, setPastPredictions] = useState([])
  const [currentView, setCurrentView] = useState('upcoming') // 'upcoming' or 'past'

  useEffect(() => {
    if (currentView === 'upcoming') {
      fetchPredictions()
    } else {
      fetchPastPredictions()
    }
  }, [currentView])

  const fetchPredictions = async () => {
    const response = await fetch("https://mmamath.onrender.com/upcoming")
    const data = await response.json()
    setPredictions(data.predictions)
  }

  const fetchPastPredictions = async () => {
    const response = await fetch("https://mmamath.onrender.com/past")
    const data = await response.json()
    setPastPredictions(data.past_predictions)
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      {/* Header Section */}
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ 
          color: '#ff0000', 
          fontSize: '3rem', 
          marginBottom: '10px',
          fontWeight: 'bold',
          textShadow: '2px 2px 4px rgba(0,0,0,0.3)'
        }}>
          ☝️🤓MMAMath
        </h1>
        <p style={{ 
          color: '#ddd', 
          fontSize: '1.2rem',
          maxWidth: '800px',
          margin: '0 auto',
          lineHeight: '1.5'
        }}>
          🚀Data-driven UFC fight predictions powered by machine learning algorithms
        </p>
      </header>

      {/* Main Content */}
      <main>
        <nav style={{ marginBottom: '20px', display: 'flex', justifyContent: 'center' }}>
          <button 
            onClick={() => setCurrentView('upcoming')}
            style={{ 
              marginRight: '10px', 
              backgroundColor: currentView === 'upcoming' ? '#ff0000' : '#333',
              color: 'white',
              padding: '12px 24px',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '1rem',
              fontWeight: 'bold',
              transition: 'all 0.3s ease'
            }}
          >
            🥊Upcoming Fights
          </button>
          <button 
            onClick={() => setCurrentView('past')}
            style={{ 
              backgroundColor: currentView === 'past' ? '#ff0000' : '#333',
              color: 'white',
              padding: '12px 24px',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '1rem',
              fontWeight: 'bold',
              transition: 'all 0.3s ease'
            }}
          >
            📊Past Predictions
          </button>
        </nav>
        
        {currentView === 'upcoming' ? 
          <PredictionList predictions={predictions} /> : 
          <PastPredictionList pastPredictions={pastPredictions} />
        }
      </main>

      {/* Info Section */}
      <section style={{ 
        marginTop: '60px',
        padding: '30px',
        backgroundColor: '#222',
        borderRadius: '8px',
        borderLeft: '4px solid #ff0000'
      }}>
        <h2 style={{ color: '#ff0000', marginBottom: '20px' }}>About MMAMath</h2>
        <p style={{ color: '#ddd', lineHeight: '1.6', marginBottom: '20px' }}>
          MMAMath uses machine learning algorithms trained on historical UFC fight data to predict fight outcomes with quantified confidence levels. Our system analyzes fighter statistics, performance metrics, and head-to-head comparisons to generate data-driven predictions.
        </p>
        <p style={{ color: '#ddd', lineHeight: '1.6', marginBottom: '20px' }}>
          Our weight class-specific algorithm processes key fighter metrics including win rates, striking differentials, takedown performance, knockout and submission records, decision outcomes, and opponent strength. Each fighter's complete statistical profile is compared against their opponent to identify advantages and predict likely outcomes.
        </p>
        <p style={{ color: '#ddd', lineHeight: '1.6', marginBottom: '20px' }}>
          Every prediction comes with a confidence percentage that indicates how certain our model is about the outcome. This transparency allows you to understand not just who we think will win, but how confident the algorithm is in that prediction based on the underlying data patterns.
        </p>
        <p style={{ color: '#ddd', lineHeight: '1.6', marginBottom: '20px' }}>
          Track our performance with detailed past predictions showing exactly what our algorithm predicted versus actual fight results. This accountability ensures you can evaluate the model's accuracy over time and understand its strengths and limitations.
        </p>
        <p style={{ color: '#ddd', lineHeight: '1.6', marginBottom: '20px' }}>
          MMAMath doesn't claim to predict the unpredictable, but rather provides you with the same statistical analysis that professional analysts use, presented in a clear, accessible format for fight fans who want to dig deeper into the numbers.
        </p>

        {/* Stats Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '20px',
          marginTop: '40px'
        }}>
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ color: '#ff0000', fontSize: '2rem', marginBottom: '5px' }}>500+</h3>
            <p style={{ color: '#ddd' }}>Active Fighters</p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ color: '#ff0000', fontSize: '2rem', marginBottom: '5px' }}>10,000+</h3>
            <p style={{ color: '#ddd' }}>Data Points</p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ color: '#ff0000', fontSize: '2rem', marginBottom: '5px' }}>71%</h3>
            <p style={{ color: '#ddd' }}>Accuracy Rate</p>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App