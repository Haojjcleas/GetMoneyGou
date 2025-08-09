import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Form, 
  Select, 
  Button, 
  Input, 
  InputNumber, 
  message, 
  Table, 
  Tag, 
  Space, 
  Row,
  Col,
  Tooltip
} from 'antd';
import { 
  PlayCircleOutlined,
  EyeOutlined,
  DeleteOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Option } = Select;

const ModelTraining = () => {
  const [form] = Form.useForm();
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [training, setTraining] = useState(false);
  const [loading, setLoading] = useState(false);

  const algorithms = [
    { value: 'logistic_regression', label: '逻辑回归', description: '简单快速，适合线性可分数据' },
    { value: 'random_forest', label: '随机森林', description: '集成学习，处理非线性关系' },
    { value: 'svm', label: '支持向量机', description: '适合高维数据，泛化能力强' },
    { value: 'gradient_boosting', label: '梯度提升', description: '强大的集成算法，预测精度高' }
  ];

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
      setModels(response.data.models || []);
    } catch (error) {
      console.error('获取模型失败:', error);
      message.error('获取模型列表失败');
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModel = async (values) => {
    if (!selectedDataset) {
      message.error('请先选择数据集');
      return;
    }

    try {
      setTraining(true);
      await axios.post(`http://localhost:5001/api/ml/datasets/${selectedDataset}/models`, {
        algorithm: values.algorithm,
        name: values.name,
        task_type: 'classification',
        test_size: values.test_size / 100,
        parameters: values.parameters || {}
      });

      message.success('模型创建成功，正在后台训练...');
      form.resetFields();
      fetchModels(selectedDataset);
    } catch (error) {
      console.error('训练失败:', error);
      message.error(error.response?.data?.error || '模型训练失败');
    } finally {
      setTraining(false);
    }
  };

  const handleDeleteModel = async (modelId) => {
    try {
      await axios.delete(`http://localhost:5001/api/ml/models/${modelId}`);
      message.success('模型删除成功');
      fetchModels(selectedDataset);
    } catch (error) {
      console.error('删除失败:', error);
      message.error('删除模型失败');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'training': return 'processing';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'completed': return '已完成';
      case 'training': return '训练中';
      case 'failed': return '失败';
      case 'pending': return '等待中';
      default: return status;
    }
  };

  const columns = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: (algorithm) => {
        const algo = algorithms.find(a => a.value === algorithm);
        return algo ? algo.label : algorithm;
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {getStatusText(status)}
        </Tag>
      )
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy) => accuracy ? `${(accuracy * 100).toFixed(2)}%` : '-'
    },
    {
      title: '创建时间',
      dataIndex: 'created_time',
      key: 'created_time',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: '训练时长',
      key: 'training_duration',
      render: (_, record) => {
        if (record.training_start_time && record.training_end_time) {
          const duration = new Date(record.training_end_time) - new Date(record.training_start_time);
          return `${Math.round(duration / 1000)}秒`;
        }
        return '-';
      }
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Tooltip title="删除模型">
            <Button 
              type="link" 
              danger 
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteModel(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 24 }}>模型训练</h2>

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={8}>
          <Card title="创建新模型" style={{ height: 'fit-content' }}>
            <Form
              form={form}
              layout="vertical"
              onFinish={handleTrainModel}
              initialValues={{
                test_size: 20
              }}
            >
              <Form.Item
                name="dataset"
                label="选择数据集"
                rules={[{ required: true, message: '请选择数据集' }]}
              >
                <Select
                  placeholder="请选择数据集"
                  onChange={setSelectedDataset}
                  showSearch
                  optionFilterProp="children"
                >
                  {datasets.map(dataset => (
                    <Option key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.total_records?.toLocaleString()} 条记录)
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item
                name="name"
                label="模型名称"
                rules={[{ required: true, message: '请输入模型名称' }]}
              >
                <Input placeholder="请输入模型名称" />
              </Form.Item>

              <Form.Item
                name="algorithm"
                label="选择算法"
                rules={[{ required: true, message: '请选择算法' }]}
              >
                <Select 
                  placeholder="请选择算法"
                  optionLabelProp="label"
                >
                  {algorithms.map(algo => (
                    <Option key={algo.value} value={algo.value} label={algo.label}>
                      <Tooltip title={algo.description} placement="right">
                        <div style={{ 
                          display: "flex", 
                          justifyContent: "space-between", 
                          alignItems: "center", 
                          maxWidth: "300px" 
                        }}>
                          <span style={{ fontWeight: "500" }}>{algo.label}</span>
                          <span style={{ 
                            fontSize: "12px", 
                            color: "#666", 
                            marginLeft: "8px", 
                            flex: "0 0 auto" 
                          }}>
                            {algo.description}
                          </span>
                        </div>
                      </Tooltip>
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item
                name="test_size"
                label="测试集比例 (%)"
                rules={[{ required: true, message: '请输入测试集比例' }]}
              >
                <InputNumber
                  min={10}
                  max={50}
                  style={{ width: '100%' }}
                  placeholder="20"
                />
              </Form.Item>

              <Form.Item>
                <Button 
                  type="primary" 
                  htmlType="submit" 
                  loading={training}
                  icon={<PlayCircleOutlined />}
                  block
                  size="large"
                >
                  {training ? '创建中...' : '开始训练'}
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          <Card 
            title="模型列表" 
            extra={
              <Tooltip title="刷新模型列表">
                <Button 
                  icon={<ReloadOutlined />}
                  onClick={() => selectedDataset && fetchModels(selectedDataset)}
                  loading={loading}
                >
                  刷新
                </Button>
              </Tooltip>
            }
          >
            <Table
              columns={columns}
              dataSource={models}
              rowKey="id"
              loading={loading}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: (total) => `共 ${total} 个模型`,
              }}
              locale={{ emptyText: selectedDataset ? '暂无模型，请创建新模型' : '请先选择数据集' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ModelTraining;
