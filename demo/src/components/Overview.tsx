import React from 'react';
import { Box, Typography, Grid, Card, CardContent, List, ListItem, ListItemText, Container } from '@mui/material';
import { motion } from 'framer-motion';

const features = [
  {
    title: '🦠 Microbiome Generation',
    description: 'Generate realistic human microbiome compositions using state-of-the-art diffusion models with hyperbolic embeddings.',
    items: [
      'Compositional data constraints',
      'Phylogenetic relationships',
      'Biological realism'
    ],
    gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
  },
  {
    title: '📊 Comprehensive Evaluation',
    description: 'Evaluate generated samples using multiple metrics designed specifically for microbiome data.',
    items: [
      'Microbiome Fréchet Distance (MFD)',
      'Alpha/Beta diversity analysis',
      'Biological validation'
    ],
    gradient: 'linear-gradient(135deg, #ec4899 0%, #f97316 100%)'
  },
  {
    title: '⏰ Temporal Prediction',
    description: 'Predict future microbiome states based on current compositions and environmental factors.',
    items: [
      'Time-series modeling',
      'Intervention effects',
      'Longitudinal analysis'
    ],
    gradient: 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
  },
  {
    title: '📈 Benchmarking Suite',
    description: 'Compare different generation methods with standardized benchmarks and statistical tests.',
    items: [
      'Performance metrics',
      'Statistical significance',
      'Publication-ready outputs'
    ],
    gradient: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
  }
];

const methodologySteps = [
  {
    title: 'Data Preprocessing',
    description: 'Transform raw microbiome data into compositional format and embed taxa in hyperbolic space',
    icon: '🔬',
    color: '#6366f1'
  },
  {
    title: 'Diffusion Training',
    description: 'Train diffusion model to learn the reverse process of adding noise to microbiome compositions',
    icon: '🧠',
    color: '#8b5cf6'
  },
  {
    title: 'Generation & Evaluation',
    description: 'Generate new samples and evaluate using specialized microbiome metrics',
    icon: '📊',
    color: '#ec4899'
  }
];

const Overview: React.FC = () => {
  return (
    <Container maxWidth="xl">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Typography
          variant="h2"
          component="h2"
          sx={{
            textAlign: 'center',
            marginBottom: '3rem',
            fontSize: { xs: '2.5rem', md: '3.5rem' },
            fontWeight: 800,
            background: 'linear-gradient(135deg, #ffffff 0%, #a855f7 50%, #6366f1 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            letterSpacing: '-0.02em',
          }}
        >
          🔬 Project Overview
        </Typography>
      </motion.div>

      <Grid container spacing={4} sx={{ marginBottom: '4rem' }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={6} key={index}>
            <motion.div
              initial={{ opacity: 0, y: 50, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ 
                duration: 0.6, 
                delay: index * 0.1,
                type: "spring",
                stiffness: 100
              }}
              whileHover={{ y: -8, scale: 1.02 }}
            >
              <Card
                className="glass-card"
                sx={{
                  height: '100%',
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '3px',
                    background: feature.gradient,
                  },
                }}
              >
                <CardContent sx={{ padding: '2.5rem' }}>
                  <Typography
                    variant="h4"
                    component="h3"
                    sx={{
                      color: 'white',
                      marginBottom: '1.5rem',
                      fontWeight: 700,
                      fontSize: '1.5rem',
                    }}
                  >
                    {feature.title}
                  </Typography>
                  
                  <Typography
                    variant="body1"
                    sx={{
                      marginBottom: '2rem',
                      color: 'rgba(255, 255, 255, 0.8)',
                      lineHeight: 1.6,
                      fontSize: '1.1rem',
                    }}
                  >
                    {feature.description}
                  </Typography>
                  
                  <List dense sx={{ padding: 0 }}>
                    {feature.items.map((item, itemIndex) => (
                      <ListItem key={itemIndex} sx={{ padding: '0.5rem 0' }}>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box
                                sx={{
                                  width: '6px',
                                  height: '6px',
                                  borderRadius: '50%',
                                  background: feature.gradient,
                                  marginRight: '1rem',
                                  flexShrink: 0,
                                }}
                              />
                              <Typography
                                sx={{
                                  color: 'rgba(255, 255, 255, 0.9)',
                                  fontWeight: 500,
                                }}
                              >
                                {item}
                              </Typography>
                            </Box>
                          }
                          sx={{ margin: 0 }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        ))}
      </Grid>

      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Card className="glass-card" sx={{ padding: '3rem' }}>
          <Typography
            variant="h3"
            component="h3"
            sx={{
              color: 'white',
              marginBottom: '2rem',
              fontWeight: 700,
              fontSize: { xs: '2rem', md: '2.5rem' },
              textAlign: 'center',
            }}
          >
            🧬 Methodology
          </Typography>
          
          <Typography
            variant="h6"
            sx={{
              marginBottom: '3rem',
              color: 'rgba(255, 255, 255, 0.8)',
              lineHeight: 1.7,
              textAlign: 'center',
              maxWidth: '800px',
              margin: '0 auto 3rem',
              fontSize: '1.2rem',
              fontWeight: 400,
            }}
          >
            Our approach combines diffusion models with hyperbolic embeddings to capture the complex 
            hierarchical structure of microbial communities. The model learns to generate compositionally 
            valid microbiome samples that preserve both phylogenetic relationships and biological constraints.
          </Typography>

          <Grid container spacing={4}>
            {methodologySteps.map((step, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.6 + index * 0.2 }}
                  whileHover={{ y: -5 }}
                >
                  <Box
                    sx={{
                      textAlign: 'center',
                      padding: '2rem',
                      background: 'rgba(255, 255, 255, 0.03)',
                      borderRadius: '20px',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: '2px',
                        background: `linear-gradient(90deg, ${step.color}, ${step.color}CC)`,
                      },
                    }}
                  >
                    <Box
                      sx={{
                        width: '80px',
                        height: '80px',
                        borderRadius: '50%',
                        background: `linear-gradient(135deg, ${step.color}40, ${step.color}20)`,
                        border: `2px solid ${step.color}60`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '2rem',
                        marginBottom: '1.5rem',
                        backdropFilter: 'blur(10px)',
                      }}
                    >
                      {step.icon}
                    </Box>
                    
                    <Typography
                      variant="h5"
                      sx={{
                        color: 'white',
                        fontWeight: 600,
                        marginBottom: '1rem',
                        fontSize: '1.3rem',
                      }}
                    >
                      {step.title}
                    </Typography>
                    
                    <Typography
                      variant="body2"
                      sx={{
                        color: 'rgba(255, 255, 255, 0.7)',
                        lineHeight: 1.6,
                        fontSize: '1rem',
                      }}
                    >
                      {step.description}
                    </Typography>
                  </Box>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Card>
      </motion.div>
    </Container>
  );
};

export default Overview;