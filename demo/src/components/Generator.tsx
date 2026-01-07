import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
} from '@mui/material';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

interface GenerationParams {
  numSamples: number;
  diversityLevel: 'low' | 'medium' | 'high';
  sampleType: 'gut' | 'oral' | 'skin';
}

interface GeneratedSample {
  name: string;
  value: number;
  color: string;
}

interface DiversityMetrics {
  shannon: number;
  simpson: number;
  observedTaxa: number;
}

const COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22'];

const TAXA_NAMES = [
  'Bacteroides', 'Firmicutes', 'Proteobacteria', 'Actinobacteria',
  'Verrucomicrobia', 'Fusobacteria', 'Cyanobacteria', 'Spirochaetes'
];

const Generator: React.FC = () => {
  const [params, setParams] = useState<GenerationParams>({
    numSamples: 10,
    diversityLevel: 'medium',
    sampleType: 'gut',
  });
  
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedData, setGeneratedData] = useState<GeneratedSample[]>([]);
  const [diversityMetrics, setDiversityMetrics] = useState<DiversityMetrics | null>(null);

  const generateMockData = (): GeneratedSample[] => {
    const diversityFactors = {
      low: { alpha: 0.5, numTaxa: 4 },
      medium: { alpha: 1.0, numTaxa: 6 },
      high: { alpha: 2.0, numTaxa: 8 }
    };
    
    const factor = diversityFactors[params.diversityLevel];
    const selectedTaxa = TAXA_NAMES.slice(0, factor.numTaxa);
    
    // Generate Dirichlet-like distribution
    const rawValues = selectedTaxa.map(() => Math.pow(Math.random(), 1 / factor.alpha));
    const sum = rawValues.reduce((a, b) => a + b, 0);
    
    return selectedTaxa.map((taxon, index) => ({
      name: taxon,
      value: rawValues[index] / sum,
      color: COLORS[index % COLORS.length],
    }));
  };

  const calculateDiversityMetrics = (data: GeneratedSample[]): DiversityMetrics => {
    const abundances = data.map(d => d.value);
    
    // Shannon diversity
    const shannon = -abundances.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
    
    // Simpson index
    const simpson = abundances.reduce((sum, p) => sum + p * p, 0);
    
    // Observed taxa
    const observedTaxa = abundances.filter(p => p > 0.001).length;
    
    return { shannon, simpson, observedTaxa };
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const data = generateMockData();
    const metrics = calculateDiversityMetrics(data);
    
    setGeneratedData(data);
    setDiversityMetrics(metrics);
    setIsGenerating(false);
  };

  return (
    <Box>
      <Typography
        variant="h3"
        component="h2"
        sx={{
          textAlign: 'center',
          marginBottom: '2rem',
          color: 'secondary.main',
          fontWeight: 600,
        }}
      >
        🎲 Generate Microbiome Samples
      </Typography>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card sx={{ marginBottom: '2rem', background: '#f8f9fa' }}>
          <CardContent sx={{ padding: '2rem' }}>
            <Grid container spacing={3} alignItems="flex-end">
              <Grid item xs={12} md={3}>
                <Typography gutterBottom>Number of Samples: {params.numSamples}</Typography>
                <Slider
                  value={params.numSamples}
                  onChange={(_, value) => setParams({ ...params, numSamples: value as number })}
                  min={1}
                  max={100}
                  valueLabelDisplay="auto"
                  sx={{ color: 'primary.main' }}
                />
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Diversity Level</InputLabel>
                  <Select
                    value={params.diversityLevel}
                    label="Diversity Level"
                    onChange={(e) => setParams({ ...params, diversityLevel: e.target.value as any })}
                  >
                    <MenuItem value="low">Low Diversity</MenuItem>
                    <MenuItem value="medium">Medium Diversity</MenuItem>
                    <MenuItem value="high">High Diversity</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Sample Type</InputLabel>
                  <Select
                    value={params.sampleType}
                    label="Sample Type"
                    onChange={(e) => setParams({ ...params, sampleType: e.target.value as any })}
                  >
                    <MenuItem value="gut">Gut Microbiome</MenuItem>
                    <MenuItem value="oral">Oral Microbiome</MenuItem>
                    <MenuItem value="skin">Skin Microbiome</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Button
                  variant="contained"
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  fullWidth
                  sx={{
                    padding: '1rem 2rem',
                    borderRadius: '25px',
                    fontWeight: 600,
                    background: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 6px 20px rgba(52, 152, 219, 0.4)',
                    },
                  }}
                  startIcon={isGenerating ? <CircularProgress size={20} color="inherit" /> : null}
                >
                  {isGenerating ? 'Generating...' : 'Generate Samples'}
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </motion.div>

      {generatedData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom textAlign="center">
                    Generated Composition
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <PieChart>
                      <Pie
                        data={generatedData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, value }) => `${name}: ${(value * 100).toFixed(1)}%`}
                        outerRadius={120}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {generatedData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom textAlign="center">
                    Diversity Metrics
                  </Typography>
                  {diversityMetrics && (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid #e9ecef' }}>
                        <Typography fontWeight={600}>Shannon Diversity:</Typography>
                        <Typography color="primary.main" fontWeight="bold">
                          {diversityMetrics.shannon.toFixed(3)}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid #e9ecef' }}>
                        <Typography fontWeight={600}>Simpson Index:</Typography>
                        <Typography color="primary.main" fontWeight="bold">
                          {diversityMetrics.simpson.toFixed(3)}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0' }}>
                        <Typography fontWeight={600}>Observed Taxa:</Typography>
                        <Typography color="primary.main" fontWeight="bold">
                          {diversityMetrics.observedTaxa}
                        </Typography>
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </motion.div>
      )}
    </Box>
  );
};

export default Generator;