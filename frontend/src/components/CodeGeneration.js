import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Select, 
  Button, 
  message, 
  Table, 
  Tag, 
  Space, 
  Modal,
  Row,
  Col,
  Typography,
  Tooltip,
  Divider,
  Input
} from 'antd';
import { 
  CodeOutlined, 
  DownloadOutlined,
  EditOutlined,
  CopyOutlined,
  BugOutlined,
  SettingOutlined
} from '@ant-design/icons';
import Editor from '@monaco-editor/react';
import axios from 'axios';

const { Option } = Select;
const { TextArea } = Input;

const CodeGeneration = () => {
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generatedCodes, setGeneratedCodes] = useState([]);
  const [editorVisible, setEditorVisible] = useState(false);
  const [currentCode, setCurrentCode] = useState(null);
  const [editorContent, setEditorContent] = useState('');
  const [testing, setTesting] = useState(false);
  const [explanationModalVisible, setExplanationModalVisible] = useState(false);
  const [customExplanation, setCustomExplanation] = useState('');
  const [currentLanguage, setCurrentLanguage] = useState('');

  const languages = [
    { value: 'python', label: 'Python', extension: '.py', mode: 'python' },
    { value: 'cpp', label: 'C++', extension: '.cpp', mode: 'cpp' }
  ];

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      fetchModels(selectedDataset);
    }
  }, [selectedDataset]);

  useEffect(() => {
    if (selectedModel) {
      fetchGeneratedCodes(selectedModel);
    }
  }, [selectedModel]);

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

  const fetchGeneratedCodes = async (modelId) => {
    try {
      const response = await axios.get(`http://localhost:5001/api/codegen/models/${modelId}/codes`);
      setGeneratedCodes(response.data.codes || []);
    } catch (error) {
      console.error('获取生成代码失败:', error);
      message.error('获取生成代码列表失败');
    }
  };

  const handleGenerateCode = async (language, useCustomExplanation = false) => {
    if (!selectedModel) {
      message.error('请先选择一个已完成训练的模型');
      return;
    }

    try {
      setGenerating(true);
      const requestData = { language: language };
      
      if (useCustomExplanation && customExplanation.trim()) {
        requestData.explanation = customExplanation.trim();
      }
      
      const response = await axios.post(`http://localhost:5001/api/codegen/models/${selectedModel}/generate`, requestData);

      message.success(`${language === 'python' ? 'Python' : 'C++'}代码生成成功！`);
      fetchGeneratedCodes(selectedModel);
      
      if (useCustomExplanation) {
        setExplanationModalVisible(false);
        setCustomExplanation('');
      }
    } catch (error) {
      console.error('代码生成失败:', error);
      message.error(error.response?.data?.error || '代码生成失败');
    } finally {
      setGenerating(false);
    }
  };

  const showExplanationModal = (language) => {
    setCurrentLanguage(language);
    setCustomExplanation('');
    setExplanationModalVisible(true);
  };

  const handleEditCode = async (codeId) => {
    try {
      const response = await axios.get(`http://localhost:5001/api/codegen/codes/${codeId}`);
      setCurrentCode(response.data);
      setEditorContent(response.data.code_content);
      setEditorVisible(true);
    } catch (error) {
      console.error('获取代码详情失败:', error);
      message.error('获取代码详情失败');
    }
  };

  const handleSaveCode = async () => {
    if (!currentCode) return;

    try {
      await axios.put(`http://localhost:5001/api/codegen/codes/${currentCode.id}`, {
        code_content: editorContent
      });
      message.success('代码保存成功');
      setEditorVisible(false);
      fetchGeneratedCodes(selectedModel);
    } catch (error) {
      console.error('保存代码失败:', error);
      message.error('保存代码失败');
    }
  };

  const handleTestCode = async (codeId) => {
    try {
      setTesting(true);
      const response = await axios.post(`http://localhost:5001/api/codegen/codes/${codeId}/test`, {
        test_data: []
      });

      const testResult = response.data.test_result;
      const status = testResult.status === 'success' ? '✅ 成功' : '❌ 失败';
      
      Modal.info({
        title: '代码测试结果',
        content: (
          <div>
            <p><strong>状态:</strong> {status}</p>
            <p><strong>消息:</strong> {testResult.message}</p>
            <p><strong>执行时间:</strong> {testResult.execution_time}s</p>
            <p><strong>内存使用:</strong> {testResult.memory_usage}</p>
            <p><strong>测试通过率:</strong> {testResult.summary.success_rate}%</p>
          </div>
        ),
        width: 600
      });
    } catch (error) {
      console.error('代码测试失败:', error);
      message.error('代码测试失败');
    } finally {
      setTesting(false);
    }
  };

  const handleDownloadCode = async (codeId) => {
    try {
      const response = await axios.get(`http://localhost:5001/api/codegen/codes/${codeId}/download`, {
        responseType: 'blob'
      });
      
      const code = generatedCodes.find(c => c.id === codeId);
      const language = languages.find(l => l.value === code.language);
      const filename = `model_${selectedModel}_${code.language}_code${language.extension}`;
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      message.success('代码下载成功');
    } catch (error) {
      console.error('下载失败:', error);
      message.error('下载代码失败');
    }
  };

  const handleCopyCode = (code) => {
    navigator.clipboard.writeText(code.code_content).then(() => {
      message.success('代码已复制到剪贴板');
    }).catch(() => {
      message.error('复制失败');
    });
  };

  const columns = [
    {
      title: '语言',
      dataIndex: 'language',
      key: 'language',
      render: (language) => {
        const lang = languages.find(l => l.value === language);
        return <Tag color="blue">{lang ? lang.label : language}</Tag>;
      }
    },
    {
      title: '生成时间',
      dataIndex: 'generated_time',
      key: 'generated_time',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: '测试状态',
      dataIndex: 'is_tested',
      key: 'is_tested',
      render: (tested) => (
        <Tag color={tested ? 'success' : 'default'}>
          {tested ? '已测试' : '未测试'}
        </Tag>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Tooltip title="编辑代码">
            <Button 
              type="link" 
              icon={<EditOutlined />}
              onClick={() => handleEditCode(record.id)}
            />
          </Tooltip>
          <Tooltip title="测试代码">
            <Button 
              type="link" 
              icon={<BugOutlined />}
              onClick={() => handleTestCode(record.id)}
              loading={testing}
            />
          </Tooltip>
          <Tooltip title="复制代码">
            <Button 
              type="link" 
              icon={<CopyOutlined />}
              onClick={() => handleCopyCode(record)}
            />
          </Tooltip>
          <Tooltip title="下载代码">
            <Button 
              type="link" 
              icon={<DownloadOutlined />}
              onClick={() => handleDownloadCode(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 24 }}>代码生成</h2>

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
          style={{ width: "100%", marginTop: 8 }}
          placeholder="请选择已完成的模型"
          onChange={setSelectedModel}
          value={selectedModel}
          loading={loading}
          disabled={!selectedDataset}
          optionLabelProp="label"
        >
          {models.map(model => (
            <Option key={model.id} value={model.id} label={model.name}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", maxWidth: "400px" }}>
                <span style={{ fontWeight: "500" }}>{model.name}</span>
                <span style={{ fontSize: "12px", color: "#666", marginLeft: "8px", flex: "0 0 auto" }}>
                  {model.algorithm} - 准确率: {(model.accuracy * 100).toFixed(1)}%
                </span>
              </div>
            </Option>
          ))}
        </Select>
              </div>
              <Divider />

              <div>
                <Space direction="vertical" style={{ width: '100%' }}>
                  {languages.map(lang => (
                    <div key={lang.value} style={{ display: 'flex', gap: '8px' }}>
                      <Button 
                        icon={<CodeOutlined />}
                        onClick={() => handleGenerateCode(lang.value)}
                        loading={generating}
                        disabled={!selectedModel}
                        style={{ flex: 1 }}
                      >
                        生成 {lang.label}
                      </Button>
                      <Tooltip title="使用自定义解释生成代码">
                        <Button 
                          icon={<SettingOutlined />}
                          onClick={() => showExplanationModal(lang.value)}
                          disabled={!selectedModel}
                        />
                      </Tooltip>
                    </div>
                  ))}
                </Space>
              </div>
            </Space>
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          <Card title="生成的代码">
            <Table
              columns={columns}
              dataSource={generatedCodes}
              rowKey="id"
              pagination={{
                pageSize: 10,
                showTotal: (total) => `共 ${total} 个代码文件`,
              }}
              locale={{ emptyText: selectedModel ? '暂无生成的代码，请点击生成按钮' : '请先选择模型' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 自定义解释弹窗 */}
      <Modal
        title="自定义AI解释"
        open={explanationModalVisible}
        onCancel={() => setExplanationModalVisible(false)}
        onOk={() => handleGenerateCode(currentLanguage, true)}
        okText="生成代码"
        cancelText="取消"
        width={800}
      >
        <div style={{ marginBottom: 16 }}>
          <p>您可以输入自定义的模型解释，生成的代码将基于这个解释进行优化：</p>
        </div>
        <TextArea
          rows={10}
          value={customExplanation}
          onChange={(e) => setCustomExplanation(e.target.value)}
          placeholder="请输入模型的预测逻辑解释、特征重要性分析等内容..."
        />
      </Modal>

      {/* 代码编辑器弹窗 */}
      <Modal
        title={`编辑 ${currentCode ? languages.find(l => l.value === currentCode.language)?.label : ''} 代码`}
        open={editorVisible}
        onCancel={() => setEditorVisible(false)}
        width="90%"
        style={{ top: 20 }}
        footer={[
          <Button key="cancel" onClick={() => setEditorVisible(false)}>
            取消
          </Button>,
          <Button key="save" type="primary" onClick={handleSaveCode}>
            保存代码
          </Button>,
        ]}
      >
        <div style={{ height: '70vh' }}>
          <Editor
            height="100%"
            language={currentCode ? languages.find(l => l.value === currentCode.language)?.mode : 'python'}
            value={editorContent}
            onChange={setEditorContent}
            theme="vs-dark"
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              wordWrap: 'on',
              automaticLayout: true,
            }}
          />
        </div>
      </Modal>
    </div>
  );
};

export default CodeGeneration;
