import React from 'react';
import { Box, Button, Container, Chip } from '@mui/material';
import { motion } from 'framer-motion';

export type Section = 'overview' | 'generator' | 'evaluation' | 'comparison' | 'temporal';

interface NavigationProps {
  activeSection: Section;
  onSectionChange: (section: Section) => void;
}

const navigationItems = [
  { key: 'overview' as Section, label: 'Overview', icon: '🔬', color: '#6366f1' },
  { key: 'generator' as Section, label: 'Generate', icon: '🎲', color: '#8b5cf6' },
  { key: 'evaluation' as Section, label: 'Evaluate', icon: '📊', color: '#ec4899' },
  { key: 'comparison' as Section, label: 'Compare', icon: '⚖️', color: '#f59e0b' },
  { key: 'temporal' as Section, label: 'Predict', icon: '⏰', color: '#10b981' },
];

const Navigation: React.FC<NavigationProps> = ({ activeSection, onSectionChange }) => {
  return (
    <Box
      component={motion.nav}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.3 }}
      className="nav-container"
      sx={{
        padding: '1.5rem 0',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            gap: { xs: '0.5rem', md: '1rem' },
            flexWrap: 'wrap',
            alignItems: 'center',
          }}
        >
          {navigationItems.map((item, index) => (
            <motion.div
              key={item.key}
              initial={{ opacity: 0, y: 20, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ 
                duration: 0.5, 
                delay: index * 0.1 + 0.4,
                type: "spring",
                stiffness: 100
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                variant={activeSection === item.key ? 'contained' : 'outlined'}
                onClick={() => onSectionChange(item.key)}
                sx={{
                  borderRadius: '16px',
                  padding: { xs: '8px 16px', md: '12px 24px' },
                  fontWeight: 600,
                  fontSize: { xs: '0.85rem', md: '1rem' },
                  minWidth: { xs: 'auto', md: '140px' },
                  height: '48px',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  background: activeSection === item.key 
                    ? `linear-gradient(135deg, ${item.color} 0%, ${item.color}CC 100%)`
                    : 'rgba(255, 255, 255, 0.05)',
                  border: activeSection === item.key 
                    ? 'none'
                    : '1px solid rgba(255, 255, 255, 0.1)',
                  color: activeSection === item.key ? 'white' : 'rgba(255, 255, 255, 0.8)',
                  backdropFilter: 'blur(20px)',
                  boxShadow: activeSection === item.key 
                    ? `0 8px 32px ${item.color}40`
                    : 'none',
                  '&:hover': {
                    background: activeSection === item.key 
                      ? `linear-gradient(135deg, ${item.color} 0%, ${item.color}DD 100%)`
                      : 'rgba(255, 255, 255, 0.1)',
                    borderColor: activeSection === item.key 
                      ? 'transparent'
                      : 'rgba(255, 255, 255, 0.2)',
                    transform: 'translateY(-2px)',
                    boxShadow: activeSection === item.key 
                      ? `0 12px 40px ${item.color}50`
                      : '0 8px 32px rgba(0, 0, 0, 0.2)',
                  },
                }}
                startIcon={
                  <Box
                    sx={{
                      fontSize: '1.2rem',
                      display: 'flex',
                      alignItems: 'center',
                      filter: activeSection === item.key ? 'none' : 'grayscale(0.3)',
                    }}
                  >
                    {item.icon}
                  </Box>
                }
              >
                <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
                  {item.label}
                </Box>
              </Button>
            </motion.div>
          ))}
        </Box>
        
        {/* Active section indicator */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            marginTop: '1rem',
          }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            key={activeSection}
          >
            <Chip
              label={navigationItems.find(item => item.key === activeSection)?.label}
              sx={{
                background: 'rgba(255, 255, 255, 0.1)',
                color: 'rgba(255, 255, 255, 0.8)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                fontSize: '0.75rem',
                height: '24px',
              }}
            />
          </motion.div>
        </Box>
      </Container>
    </Box>
  );
};

export default Navigation;