import React from 'react'
import {
  Box,
  Grid,
  Text,
  VStack,
  Progress,
  List,
  ListItem,
  ListIcon,
  Spinner,
  Badge,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  HStack,
  Divider,
  Button,
  useColorModeValue,
  useToast,
} from '@chakra-ui/react'
import {
  FaCheckCircle,
  FaMicrophone,
  FaChartLine,
  FaSmile,
  FaExclamationTriangle,
  FaStar,
  FaLightbulb,
  FaDownload,
} from 'react-icons/fa'
import api from '../config/api'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts'

const getColorScheme = (score: number): string => {
  if (score >= 80) return 'green'
  if (score >= 60) return 'blue'
  if (score >= 40) return 'yellow'
  return 'red'
}

const normalizeValue = (value: number): number => {
  return Math.min(100, Math.max(0, value * 100))
}

export const AnalysisDisplay: React.FC<AnalysisDisplayProps> = ({ analysis }) => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const cardBg = useColorModeValue('gray.50', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const textColor = useColorModeValue('gray.700', 'gray.200');
  const secondaryTextColor = useColorModeValue('gray.600', 'gray.400');
  const toast = useToast();

  const metrics = React.useMemo(() => {
    if (!analysis?.audio_metrics?.clarity?.clarity_score || 
        !analysis?.audio_metrics?.clarity?.pronunciation_score ||
        !analysis?.audio_metrics?.rhythm?.speech_rate ||
        !analysis?.audio_metrics?.pitch?.pitch_variation) {
      return []
    }
    
    const { clarity, pitch, rhythm } = analysis.audio_metrics
    
    return [
      {
        name: 'Clarity',
        value: normalizeValue(clarity.clarity_score),
        color: 'blue.400',
      },
      {
        name: 'Pronunciation',
        value: normalizeValue(clarity.pronunciation_score),
        color: 'green.400',
      },
      {
        name: 'Speech Rate',
        value: normalizeValue(rhythm.speech_rate),
        color: 'purple.400',
      },
      {
        name: 'Pitch Control',
        value: normalizeValue(pitch.pitch_variation),
        color: 'orange.400',
      },
    ]
  }, [analysis])

  const emotionMetrics = React.useMemo(() => {
    if (!analysis?.audio_metrics?.emotion?.emotion_scores) return []
    
    return Object.entries(analysis.audio_metrics.emotion.emotion_scores).map(([emotion, score]) => ({
      emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      score: normalizeValue(score),
    }))
  }, [analysis])

  if (!analysis) {
    return (
      <Box w="full" h="400px" display="flex" alignItems="center" justifyContent="center">
        <Spinner size="xl" color="purple.500" />
      </Box>
    )
  }

  const dominantEmotion = analysis.audio_metrics?.emotion?.dominant_emotion || 'Unknown'
  const emotionConfidence = analysis.audio_metrics?.emotion?.emotion_confidence || 0
  const sessionType = analysis.metadata?.session_type || 'Unknown'
  const duration = analysis.metadata?.duration || 0
  const averagePitch = analysis.audio_metrics?.pitch?.average_pitch || 0
  const keyPoints = analysis.recommendations?.key_points || []
  const improvementAreas = analysis.recommendations?.improvement_areas || []
  const practiceSuggestions = analysis.recommendations?.practice_suggestions || []
  const strengths = analysis.recommendations?.strengths || []

  const handleDownloadPDF = async () => {
    try {
      const response = await api.post('/export-analysis-pdf', analysis, {
        responseType: 'blob',
      });
      
      // Create blob link to download
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      // Get filename from response headers or create default
      const contentDisposition = response.headers['content-disposition'];
      let filename = 'voice_analysis_report.pdf';
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="(.+)"/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast({
        title: 'PDF Downloaded',
        description: 'Your voice analysis report has been downloaded successfully!',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error downloading PDF:', error);
      toast({
        title: 'Download Failed',
        description: 'Failed to download PDF. Please try again.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  return (
    <Box w="full" bg={bgColor} p={6} borderRadius="lg" boxShadow="md" border="1px solid" borderColor={borderColor}>
      <VStack spacing={6} align="stretch">
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Text fontSize="2xl" fontWeight="bold" color={textColor}>
            Voice Analysis Results
          </Text>
          <HStack spacing={3}>
            <Button
              leftIcon={<FaDownload />}
              colorScheme="purple"
              variant="outline"
              size="sm"
              onClick={handleDownloadPDF}
              _hover={{ bg: 'purple.50' }}
            >
              Download PDF
            </Button>
            <HStack spacing={2}>
              <Badge
                colorScheme={getColorScheme(emotionConfidence * 100)}
                fontSize="md"
                px={3}
                py={1}
                borderRadius="full"
              >
                {dominantEmotion}
              </Badge>
              <Badge
                colorScheme="purple"
                variant="outline"
                fontSize="sm"
              >
                {sessionType}
              </Badge>
            </HStack>
          </HStack>
        </Box>

        <Tabs variant="soft-rounded" colorScheme="purple">
          <TabList>
            <Tab>Metrics</Tab>
            <Tab>Emotions</Tab>
            <Tab>Recommendations</Tab>
            <Tab>Transcription</Tab>
          </TabList>

          <TabPanels>
            <TabPanel>
              <Grid templateColumns={['1fr', '1fr', 'repeat(2, 1fr)']} gap={6}>
                {metrics.map((metric) => (
                  <Box key={metric.name} p={4} borderRadius="md" bg={cardBg} border="1px solid" borderColor={borderColor}>
                    <Text fontSize="lg" fontWeight="semibold" mb={2}>
                      {metric.name}
                    </Text>
                    <Progress
                      value={metric.value}
                      colorScheme={getColorScheme(metric.value)}
                      size="lg"
                      borderRadius="full"
                    />
                    <Text mt={2} fontSize="sm" color={secondaryTextColor}>
                      {metric.value.toFixed(1)}%
                    </Text>
                  </Box>
                ))}
              </Grid>

              <Box h={['200px', '250px', '300px']} mt={6}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#805AD5" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </TabPanel>

            <TabPanel>
              <Box h={['300px', '350px', '400px']}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={emotionMetrics}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="emotion" />
                    <PolarRadiusAxis domain={[0, 100]} />
                    <Radar
                      name="Emotion"
                      dataKey="score"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.6}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </Box>
            </TabPanel>

            <TabPanel>
              <Grid templateColumns={['1fr', '1fr', 'repeat(2, 1fr)']} gap={6}>
                <Box>
                  <HStack mb={3}>
                    <FaStar color="#805AD5" />
                    <Text fontSize="lg" fontWeight="semibold">
                      Key Points
                    </Text>
                  </HStack>
                  <List spacing={3}>
                    {keyPoints.map((point, index) => (
                      <ListItem key={index} display="flex" alignItems="center">
                        <ListIcon as={FaCheckCircle} color="green.500" />
                        {point}
                      </ListItem>
                    ))}
                  </List>
                </Box>

                <Box>
                  <HStack mb={3}>
                    <FaExclamationTriangle color="#805AD5" />
                    <Text fontSize="lg" fontWeight="semibold">
                      Areas for Improvement
                    </Text>
                  </HStack>
                  <List spacing={3}>
                    {improvementAreas.map((area, index) => (
                      <ListItem key={index} display="flex" alignItems="center">
                        <ListIcon as={FaExclamationTriangle} color="orange.500" />
                        {area}
                      </ListItem>
                    ))}
                  </List>
                </Box>

                <Box>
                  <HStack mb={3}>
                    <FaLightbulb color="#805AD5" />
                    <Text fontSize="lg" fontWeight="semibold">
                      Practice Suggestions
                    </Text>
                  </HStack>
                  <List spacing={3}>
                    {practiceSuggestions.map((suggestion, index) => (
                      <ListItem key={index} display="flex" alignItems="center">
                        <ListIcon as={FaLightbulb} color="yellow.500" />
                        {suggestion}
                      </ListItem>
                    ))}
                  </List>
                </Box>

                <Box>
                  <HStack mb={3}>
                    <FaCheckCircle color="#805AD5" />
                    <Text fontSize="lg" fontWeight="semibold">
                      Strengths
                    </Text>
                  </HStack>
                  <List spacing={3}>
                    {strengths.map((strength, index) => (
                      <ListItem key={index} display="flex" alignItems="center">
                        <ListIcon as={FaCheckCircle} color="green.500" />
                        {strength}
                      </ListItem>
                    ))}
                  </List>
                </Box>
              </Grid>
            </TabPanel>

            <TabPanel>
              <Box p={4} borderRadius="md" bg="gray.50" whiteSpace="pre-wrap">
                <Text fontSize="lg" fontWeight="semibold" mb={3}>
                  Full Transcription
                </Text>
                <Text fontFamily="monospace" fontSize="md">
                  {analysis.transcription?.full_text || "No transcription available."}
                </Text>
              </Box>
              <Box mt={4} p={3} bg="gray.50" borderRadius="md">
                <Text fontSize="sm" color="gray.600">
                  Word Count: {analysis.transcription?.word_count || 0}
                </Text>
              </Box>
            </TabPanel>
          </TabPanels>
        </Tabs>

        <Divider />

        <Grid templateColumns={['1fr', '1fr', 'repeat(3, 1fr)']} gap={4}>
          <Box p={3} bg="gray.50" borderRadius="md">
            <Text fontSize="sm" color="gray.600" display="flex" alignItems="center">
              <Box as={FaMicrophone} mr={2} />
              Duration: {duration.toFixed(1)}s
            </Text>
          </Box>
          <Box p={3} bg="gray.50" borderRadius="md">
            <Text fontSize="sm" color="gray.600" display="flex" alignItems="center">
              <Box as={FaChartLine} mr={2} />
              Avg. Pitch: {averagePitch.toFixed(1)} Hz
            </Text>
          </Box>
          <Box p={3} bg="gray.50" borderRadius="md">
            <Text fontSize="sm" color="gray.600" display="flex" alignItems="center">
              <Box as={FaSmile} mr={2} />
              Confidence: {(emotionConfidence * 100).toFixed(1)}%
            </Text>
          </Box>
        </Grid>
      </VStack>
    </Box>
  )
} 