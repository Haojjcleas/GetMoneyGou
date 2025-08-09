import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Select, 
  Button, 
  Input, 
  message, 
  Space, 
  Row,
  Col,
  Divider,
  Typography,
  List,
  Tooltip
} from 'antd';
import { 
  RobotOutlined, 
  SendOutlined,
  BulbOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Option } = Select;
const { Title, Paragraph, Text } = Typography;

const AIExplanation = () => {
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [explanation, setExplanation] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      fetchModels(selectedDataset);
    }
  }, [selectedDataset]);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get('http://localhost:5001/api/data/datasets');
      setDatasets(response.data.datasets || []);
    } catch (error) {
      console.error('获取数据集失败:', error);
      message.error('获取数据集列表失败');
    }
  };

  const fetchModels = async (datasetId) => {
    try {
      setLoading(true);
      const response = await axios.get(`http://localhost:5001/api/ml/datasets/${datasetId}/models`);
      const completedModels = (response.data.models || []).filter(model => model.status === 'completed');
      setModels(completedModels);
    } catch (error) {
      console.error('获取模型失败:', error);
      message.error('获取模型列表失败');
    } finally {
      setLoading(false);
    }
  };

  const handleExplainModel = async (question = null) => {
    if (!selectedModel) {
      message.error('请先选择一个已完成训练的模型');
      return;
    }

    try {
      setExplaining(true);
      const response = await axios.post(`http://localhost:5001/api/ai/models/${selectedModel}/explain`, {
        question: question
      });

      setExplanation(response.data.explanation);
      setSessionId(response.data.session_id);
      setChatHistory([]);
      message.success('AI解释生成成功！');
    } catch (error) {
      console.error('AI解释失败:', error);
      message.error(error.response?.data?.error || 'AI解释失败');
    } finally {
      setExplaining(false);
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim() || !sessionId) {
      return;
    }

    const userMessage = chatInput.trim();
    setChatInput('');
    
    // 添加用户消息到聊天历史
    const newUserMessage = {
      type: 'user',
      content: userMessage,
      timestamp: new Date().toLocaleTimeString()
    };
    setChatHistory(prev => [...prev, newUserMessage]);

    try {
      setChatLoading(true);
      const response = await axios.post(`http://localhost:5001/api/ai/models/${selectedModel}/chat`, {
        session_id: sessionId,
        question: userMessage
      });

      // 添加AI回复到聊天历史
      const aiMessage = {
        type: 'assistant',
        content: response.data.response,
        timestamp: new Date().toLocaleTimeString()
      };
      setChatHistory(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('对话失败:', error);
      message.error('AI对话失败');
      
      // 添加错误消息
      const errorMessage = {
        type: 'assistant',
        content: '抱歉，我暂时无法回答您的问题。请稍后再试。',
        timestamp: new Date().toLocaleTimeString()
      };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setChatLoading(false);
    }
  };

  const formatExplanation = (text) => {
    // 简单的Markdown格式化
    return text
      .split('\n')
      .map((line, index) => {
        if (line.startsWith('## ')) {
          return <Title key={index} level={3}>{line.substring(3)}</Title>;
        } else if (line.startsWith('### ')) {
          return <Title key={index} level={4}>{line.substring(4)}</Title>;
        } else if (line.startsWith('**') && line.endsWith('**')) {
          return <Text key={index} strong>{line.slice(2, -2)}</Text>;
        } else if (line.startsWith('- ')) {
          return <li key={index}>{line.substring(2)}</li>;
        } else if (line.trim()) {
          return <Paragraph key={index}>{line}</Paragraph>;
        }
        return <br key={index} />;
      });
  };

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 24 }}>AI模型解释</h2>

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={8}>
          <Card title="选择模型" style={{ height: 'fit-content' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <label>选择数据集：</label>
                <Select
                  style={{ width: '100%', marginTop: 8 }}
                  placeholder="请选择数据集"
                  onChange={setSelectedDataset}
                  value={selectedDataset}
                >
                  {datasets.map(dataset => (
                    <Option key={dataset.id} value={dataset.id}>
                      {dataset.name}
                    </Option>
                  ))}
                </Select>
              </div>

              <div>
                <label>选择模型：</label>
                <Select
                  style={{ width: '100%', marginTop: 8 }}
                  placeholder="请选择已完成的模型"
                  onChange={setSelectedModel}
                  value={selectedModel}
                  loading={loading}
                  disabled={!selectedDataset}
                  optionLabelProp="label"
                >
                  {models.map(model => (
                    <Option key={model.id} value={model.id} label={model.name}>
                      <div style={{ 
                        display: "flex", 
                        justifyContent: "space-between", 
                        alignItems: "center", 
                        maxWidth: "400px" 
                      }}>
                        <span style={{ fontWeight: "500" }}>{model.name}</span>
                        <span style={{ 
                          fontSize: "12px", 
                          color: "#666", 
                          marginLeft: "8px", 
                          flex: "0 0 auto" 
                        }}>
                          {model.algorithm} - 准确率: {(model.accuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                    </Option>
                  ))}
                </Select>
              </div>

              <Button 
                type="primary" 
                icon={<RobotOutlined />}
                onClick={() => handleExplainModel()}
                loading={explaining}
                disabled={!selectedModel}
                block
                size="large"
              >
                {explaining ? '生成解释中...' : '生成AI解释'}
              </Button>

              <Divider />

              <div>
                <Text strong>快速问题：</Text>
                <Space direction="vertical" style={{ width: '100%', marginTop: 8 }}>
                  <Tooltip title="分析哪些特征对预测结果影响最大">
                    <Button 
                      size="small" 
                      onClick={() => handleExplainModel('哪些特征对预测最重要？')}
                      disabled={!selectedModel || explaining}
                      block
                    >
                      <BulbOutlined /> 重要特征分析
                    </Button>
                  </Tooltip>
                  <Tooltip title="解释模型如何做出预测决策">
                    <Button 
                      size="small" 
                      onClick={() => handleExplainModel('模型是如何做出预测决策的？')}
                      disabled={!selectedModel || explaining}
                      block
                    >
                      <BarChartOutlined /> 预测逻辑解释
                    </Button>
                  </Tooltip>
                </Space>
              </div>
            </Space>
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          <Card title="AI解释结果" style={{ minHeight: 600 }}>
            {explanation ? (
              <div>
                <div style={{ 
                  maxHeight: 400, 
                  overflowY: 'auto', 
                  padding: 16, 
                  background: '#fafafa', 
                  borderRadius: 6,
                  marginBottom: 16
                }}>
                  {formatExplanation(explanation)}
                </div>

                <Divider>继续对话</Divider>

                <div style={{ marginBottom: 16 }}>
                  <List
                    dataSource={chatHistory}
                    renderItem={(item) => (
                      <List.Item style={{ 
                        justifyContent: item.type === 'user' ? 'flex-end' : 'flex-start',
                        padding: '8px 0'
                      }}>
                        <div style={{
                          maxWidth: '80%',
                          padding: '8px 12px',
                          borderRadius: 8,
                          background: item.type === 'user' ? '#1890ff' : '#f0f0f0',
                          color: item.type === 'user' ? 'white' : 'black'
                        }}>
                          <div>{item.content}</div>
                          <div style={{ 
                            fontSize: '12px', 
                            opacity: 0.7, 
                            marginTop: 4,
                            textAlign: 'right'
                          }}>
                            {item.timestamp}
                          </div>
                        </div>
                      </List.Item>
                    )}
                    style={{ maxHeight: 200, overflowY: 'auto' }}
                  />
                </div>

                <Space.Compact style={{ width: '100%' }}>
                  <Input
                    placeholder="继续提问关于模型的问题..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onPressEnter={handleSendMessage}
                    disabled={chatLoading}
                  />
                  <Button 
                    type="primary" 
                    icon={<SendOutlined />}
                    onClick={handleSendMessage}
                    loading={chatLoading}
                    disabled={!chatInput.trim()}
                  >
                    发送
                  </Button>
                </Space.Compact>
              </div>
            ) : (
              <div style={{ 
                textAlign: 'center', 
                padding: 60,
                color: '#999'
              }}>
                <RobotOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                <p>请选择模型并点击"生成AI解释"开始分析</p>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default AIExplanation;
