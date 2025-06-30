import React, { useState } from "react"

const PredictionList = ({predictions}) => {
    const [openEvents, setOpenEvents] = useState({})

    const getWinner = (prediction) => {
        if (prediction.prediction === "win") return prediction.fighterName;
        if (prediction.prediction === "loss") return prediction.opponentName;
        return "Draw";
    };

    // Group predictions by event
    const groupedPredictions = predictions.reduce((groups, prediction) => {
        const event = prediction.event || "Unknown Event";
        if (!groups[event]) {
            groups[event] = [];
        }
        groups[event].push(prediction);
        return groups;
    }, {});

    const toggleEvent = (eventName) => {
        setOpenEvents(prev => ({
            ...prev,
            [eventName]: !prev[eventName]
        }));
    };

    return (
        <div style={{ backgroundColor: '#333', color: 'white', padding: '20px' }}>
            <h2 style={{ color: 'white', marginBottom: '30px' }}>UFC Upcoming Fight Predictions</h2>
            {Object.entries(groupedPredictions).map(([eventName, eventPredictions]) => (
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
                        <h3 style={{ margin: 0, color: 'red' }}>{eventName}</h3>
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
                                <col style={{ width: '40%' }} />
                                <col style={{ width: '30%' }} />
                                <col style={{ width: '30%' }} />
                            </colgroup>
                            <thead>
                                <tr>
                                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #444', color: '#ff6b6b', fontWeight: '600' }}>Fight</th>
                                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #444', color: '#90ee90', fontWeight: '600' }}>Winner</th>
                                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #444', color: '#00ffff', fontWeight: '600' }}>Confidence</th>
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
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default PredictionList