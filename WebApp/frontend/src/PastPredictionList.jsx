import React, { useState } from "react"

const PastPredictionList = ({pastPredictions}) => {
    const [openEvents, setOpenEvents] = useState({})

    const getWinner = (prediction) => {
        if (prediction.prediction === "win") return prediction.fighterName;
        if (prediction.prediction === "loss") return prediction.opponentName;
        return "Draw";
    };

    // Group predictions by event
    const groupedPredictions = pastPredictions.reduce((groups, prediction) => {
        const event = prediction.event || "Unknown Event";
        if (!groups[event]) {
            groups[event] = [];
        }
        groups[event].push(prediction);
        return groups;
    }, {});

    // Get event names in reverse order
    const reversedEventNames = Object.keys(groupedPredictions).reverse();

    const toggleEvent = (eventName) => {
        setOpenEvents(prev => ({
            ...prev,
            [eventName]: !prev[eventName]
        }));
    };

    // Calculate accuracy for each event
    const getEventAccuracy = (eventPredictions) => {
        const correct = eventPredictions.filter(p => p.correct).length;
        const total = eventPredictions.length;
        return `${correct}/${total} (${Math.round((correct/total) * 100)}%)`;
    };

    return (
        <div style={{ backgroundColor: '#333', color: 'white', padding: '20px' }}>
            <h2 style={{ color: 'white', marginBottom: '30px' }}>UFC Past Fight Predictions</h2>
            {reversedEventNames.map((eventName) => {
                const eventPredictions = groupedPredictions[eventName];
                return (
                    <div key={eventName} style={{ 
                        marginBottom: '20px', 
                        border: '1px solid #444', 
                        borderRadius: '5px',
                        overflow: 'hidden'
                    }}>
                        <div 
                            onClick={() => toggleEvent(eventName)}
                            style={{
                                padding: '15px',
                                backgroundColor: '#222',
                                cursor: 'pointer',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                            }}
                        >
                            <div>
                                <h3 style={{ margin: 0, color: 'red' }}>{eventName}</h3>
                                <small style={{ color: '#aaa' }}>
                                    Accuracy: {getEventAccuracy(eventPredictions)}
                                </small>
                            </div>
                            <span style={{ fontSize: '18px', color: 'white' }}>
                                {openEvents[eventName] ? '▼' : '▶'}
                            </span>
                        </div>
                        
                        {/* Always render table but control visibility */}
                        <div style={{ 
                            maxHeight: openEvents[eventName] ? '1000px' : '0',
                            overflow: 'hidden',
                            transition: 'max-height 0.3s ease',
                            backgroundColor: '#222'
                        }}>
                            <table style={{ 
                                width: '100%', 
                                borderCollapse: 'collapse',
                                tableLayout: 'fixed'
                            }}>
                                <colgroup>
                                    <col style={{ width: '35%' }} />
                                    <col style={{ width: '25%' }} />
                                    <col style={{ width: '20%' }} />
                                    <col style={{ width: '20%' }} />
                                </colgroup>
                                <thead>
                                    <tr>
                                        <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #444', color: '#ff6b6b', fontWeight: '600' }}>Fight</th>
                                        <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #444', color: '#90ee90', fontWeight: '600' }}>Predicted Winner</th>
                                        <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #444', color: '#00ffff', fontWeight: '600' }}>Confidence</th>
                                        <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #444', color: '#ffd700', fontWeight: '600' }}>Result</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {eventPredictions.map((prediction) => (
                                        <tr key={prediction.id}>
                                            <td style={{ 
                                                padding: '12px', 
                                                borderBottom: '1px solid #444', 
                                                color: 'white',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis'
                                            }}>
                                                {`${prediction.fighterName} vs. ${prediction.opponentName}`}
                                            </td>
                                            <td style={{ 
                                                padding: '12px', 
                                                borderBottom: '1px solid #444', 
                                                color: 'white',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis'
                                            }}>
                                                {getWinner(prediction)}
                                            </td>
                                            <td style={{ 
                                                padding: '12px', 
                                                borderBottom: '1px solid #444', 
                                                color: 'white'
                                            }}>
                                                {prediction.confidence}
                                            </td>
                                            <td style={{ 
                                                padding: '12px', 
                                                borderBottom: '1px solid #444',
                                                color: prediction.correct ? '#4CAF50' : '#F44336',
                                                fontWeight: 'bold'
                                            }}>
                                                {prediction.correct ? '✓ Correct' : '✗ Incorrect'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default PastPredictionList