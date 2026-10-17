import { useState } from 'react';
import axios from 'axios';
import { Upload, Activity, FileText, Cpu, Brain, Download } from 'lucide-react';
import './index.css';

const API_BASE_URL = 'http://localhost:8000';

const pct = (v) => (v == null ? '-' : v >= 1 ? '100%*' : `${(v * 100).toFixed(1)}%`);

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (selectedFile) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResults(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResults(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'An error occurred during analysis.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadReport = () => {
    if (results?.report_url) {
      window.open(results.report_url, '_blank');
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Q-MED VISION</h1>
        <p className="subtitle">AI-Assisted Brain MRI Analysis</p>
      </header>

      {!results && !isAnalyzing && (
        <div className="glass-panel" style={{ maxWidth: '800px', margin: '0 auto' }}>
          <div 
            className="upload-section"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <Upload className="upload-icon" />
            <h2>Upload Brain MRI Scan</h2>
            <p className="subtitle" style={{ marginTop: '0.5rem' }}>Drag and drop or click to browse (JPG, PNG, DICOM)</p>
            <input 
              id="file-upload" 
              type="file" 
              accept=".jpg,.jpeg,.png,.dcm" 
              style={{ display: 'none' }}
              onChange={handleFileInput}
            />
          </div>

          {previewUrl && (
            <div style={{ textAlign: 'center' }}>
              <img src={previewUrl} alt="Preview" style={{ maxWidth: '300px', borderRadius: '8px', marginBottom: '1rem', border: '1px solid var(--glass-border)' }} />
              <div>
                <button className="btn" onClick={handleAnalyze}>
                  <Brain size={20} />
                  Analyze Scan
                </button>
              </div>
            </div>
          )}
          
          {error && <p style={{ color: 'var(--danger)', textAlign: 'center', marginTop: '1rem' }}>{error}</p>}
        </div>
      )}

      {isAnalyzing && (
        <div className="glass-panel" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
          <div className="loading-spinner"></div>
          <h2 style={{ marginTop: '2rem', color: 'var(--accent-primary)' }}>Analyzing Scan...</h2>
          <p className="subtitle">Running image quality check and tumour classification</p>
        </div>
      )}

      {results && !isAnalyzing && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
             <button className="btn" onClick={() => { setFile(null); setPreviewUrl(null); setResults(null); }} style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)' }}>
               New Scan
             </button>
             <button className="btn" onClick={handleDownloadReport}>
               <Download size={20} />
               Download PDF Report
             </button>
          </div>

          <div className="dashboard-grid">
            {/* Diagnosis & Quality */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              
              <div className="glass-panel">
                <h3 className="card-title"><Activity /> AI Analysis Result</h3>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <div className="metric-label">Prediction</div>
                    <div className={`metric-value ${results.inference.prediction.includes('Detected') && !results.inference.prediction.includes('No') ? 'high' : 'low'}`}>
                      {results.inference.prediction}
                    </div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">Probability</div>
                    <div className="metric-value">{results.inference.probability}</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">Confidence</div>
                    <div className="metric-value">{results.inference.confidence}</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">Risk Level</div>
                    <div className={`metric-value ${results.inference.risk_level.includes('HIGH') ? 'high' : results.inference.risk_level.includes('LOW') ? 'low' : 'medium'}`} style={{ fontSize: '1.2rem' }}>
                      {results.inference.risk_level}
                    </div>
                  </div>
                </div>
              </div>

              <div className="glass-panel">
                <h3 className="card-title"><FileText /> Image Quality</h3>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <span className="subtitle">Overall Status:</span>
                  <strong className={results.quality.overall_quality === 'GOOD' ? 'metric-value low' : 'metric-value high'}>
                    {results.quality.overall_quality}
                  </strong>
                </div>
                {results.quality.details && (
                  <p style={{ color: 'var(--warning)', marginBottom: '1rem', fontSize: '0.9rem' }}>{results.quality.details}</p>
                )}
                <div className="metrics-grid" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
                   <div className="metric-item" style={{ padding: '0.5rem' }}>
                    <div className="metric-label">Sharpness</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{results.quality.sharpness}</div>
                  </div>
                  <div className="metric-item" style={{ padding: '0.5rem' }}>
                    <div className="metric-label">Contrast</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{results.quality.contrast}</div>
                  </div>
                  <div className="metric-item" style={{ padding: '0.5rem' }}>
                    <div className="metric-label">Noise</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{results.quality.noise_level}</div>
                  </div>
                </div>
              </div>

            </div>

            {/* Explainability & Model Comparison */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              
              <div className="glass-panel">
                 <h3 className="card-title"><Brain /> Explainable AI (Grad-CAM)</h3>
                 <p className="subtitle" style={{ fontSize: '0.9rem' }}>Highlighted regions indicate areas of high model attention.</p>
                 <div className="images-container">
                    <div className="image-wrapper">
                      <img src={previewUrl} alt="Original MRI" />
                      <div className="image-label">Original MRI</div>
                    </div>
                    {results.inference.heatmap_url ? (
                      <div className="image-wrapper">
                        <img src={results.inference.heatmap_url} alt="AI Heatmap" style={{ width: '100%', aspectRatio: '1', borderRadius: '8px', border: '1px solid var(--glass-border)' }} />
                        <div className="image-label">AI Heatmap</div>
                      </div>
                    ) : (
                      <div className="image-wrapper">
                         <div style={{ width: '100%', aspectRatio: '1', background: 'url(https://via.placeholder.com/224x224/1e293b/8b5cf6?text=Grad-CAM+Heatmap)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}></div>
                         <div className="image-label">AI Heatmap (Placeholder)</div>
                      </div>
                    )}
                 </div>
              </div>

              {results.inference.evaluation && (
                <div className="glass-panel">
                   <h3 className="card-title"><Cpu /> Model Comparison (held-out test set)</h3>
                   <p className="subtitle" style={{ fontSize: '0.85rem', marginBottom: '1rem' }}>
                     Measured on {results.inference.evaluation.test_images} unseen MRI images. Active model: {results.inference.model_name}.
                   </p>
                   <table className="comparison-table">
                     <thead>
                       <tr>
                         <th>Model</th>
                         <th>Accuracy</th>
                         <th>Sensitivity</th>
                         <th>Specificity</th>
                         <th>F1</th>
                         <th>AUC</th>
                       </tr>
                     </thead>
                     <tbody>
                       {results.inference.evaluation.models.map((m) => (
                         <tr key={m.model} className={m.deployed ? 'quantum-highlight' : ''}>
                           <td>{m.model}{m.deployed ? ' (active)' : ''}</td>
                           <td>{pct(m.accuracy)}</td>
                           <td>{pct(m.recall_sensitivity)}</td>
                           <td>{pct(m.specificity)}</td>
                           <td>{pct(m.f1)}</td>
                           <td>{m.roc_auc == null ? '-' : m.roc_auc.toFixed(3)}</td>
                         </tr>
                       ))}
                     </tbody>
                   </table>
                   <p className="subtitle" style={{ fontSize: '0.8rem', marginTop: '0.75rem' }}>
                     * 0 errors on this test set. This is not real-world accuracy: the test set overlaps with the training data, and the model has not been validated on scans from other sources.
                   </p>
                </div>
              )}

            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
