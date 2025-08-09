import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Upload, 
  Button, 
  Form, 
  Input, 
  message, 
  Table, 
  Tag, 
  Space, 
  Divider,
  Progress,
  Alert
} from 'antd';
import { 
  InboxOutlined, 
  UploadOutlined, 
  FileTextOutlined,
  DeleteOutlined,
  EyeOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Dragger } = Upload;
const { TextArea } = Input;

const DataUpload = () => {
  const [form] = Form.useForm();
  const [uploading, setUploading] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fileList, setFileList] = useState([]);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5001/api/data/datasets');
      setDatasets(response.data.datasets || []);
    } catch (error) {
      console.error('获取数据集失败:', error);
      message.error('获取数据集列表失败');
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (values) => {
    if (fileList.length === 0) {
      message.error('请选择要上传的CSV文件');
      return;
    }

    const formData = new FormData();
    formData.append('file', fileList[0]);
    formData.append('name', values.name);
    formData.append('description', values.description || '');

    try {
      setUploading(true);
      const response = await axios.post('http://localhost:5001/api/data/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      message.success('文件上传成功！');
      form.resetFields();
      setFileList([]);
      fetchDatasets(); // 刷新数据集列表
    } catch (error) {
      console.error('上传失败:', error);
      message.error(error.response?.data?.error || '文件上传失败');
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDataset = async (datasetId) => {
    try {
      await axios.delete(`http://localhost:5001/api/data/datasets/${datasetId}`);
      message.success('数据集删除成功');
      fetchDatasets();
    } catch (error) {
      console.error('删除失败:', error);
      message.error('删除数据集失败');
    }
  };

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv',
    fileList: fileList,
    beforeUpload: (file) => {
      const isCSV = file.type === 'text/csv' || file.name.endsWith('.csv');
      if (!isCSV) {
        message.error('只能上传CSV文件！');
        return false;
      }
      const isLt100M = file.size / 1024 / 1024 < 100;
      if (!isLt100M) {
        message.error('文件大小不能超过100MB！');
        return false;
      }
      setFileList([file]);
      return false; // 阻止自动上传
    },
    onRemove: () => {
      setFileList([]);
    },
  };

  const columns = [
    {
      title: '数据集名称',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      ellipsis: true,
    },
    {
      title: '记录数',
      dataIndex: 'total_records',
      key: 'total_records',
      render: (count) => count?.toLocaleString() || 0,
    },
    {
      title: '时间范围',
      key: 'date_range',
      render: (_, record) => {
        if (record.date_range_start && record.date_range_end) {
          return (
            <div>
              <div>{new Date(record.date_range_start).toLocaleDateString()}</div>
              <div>至 {new Date(record.date_range_end).toLocaleDateString()}</div>
            </div>
          );
        }
        return '-';
      },
    },
    {
      title: '上传时间',
      dataIndex: 'upload_time',
      key: 'upload_time',
      render: (time) => new Date(time).toLocaleString(),
    },
    {
      title: '状态',
      key: 'status',
      render: () => <Tag color="success">已上传</Tag>,
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            type="link" 
            icon={<EyeOutlined />}
            onClick={() => {
              // 这里可以添加查看数据集详情的功能
              message.info('查看功能开发中...');
            }}
          >
            查看
          </Button>
          <Button 
            type="link" 
            danger 
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteDataset(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 24 }}>数据上传</h2>
      
      <Alert
        message="数据格式要求"
        description="请上传包含以下列的CSV文件：timestamp（时间戳）、open（开盘价）、high（最高价）、low（最低价）、close（收盘价）、volume（成交量）、buy_amount（买入金额）、sell_amount（卖出金额）"
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Card title="上传新数据集" style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUpload}
        >
          <Form.Item
            name="name"
            label="数据集名称"
            rules={[{ required: true, message: '请输入数据集名称' }]}
          >
            <Input placeholder="请输入数据集名称" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述（可选）"
          >
            <TextArea 
              rows={3} 
              placeholder="请描述数据集的内容和来源"
            />
          </Form.Item>

          <Form.Item
            label="CSV文件"
            required
          >
            <Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">
                支持单个CSV文件上传，文件大小不超过100MB
              </p>
            </Dragger>
          </Form.Item>

          <Form.Item>
            <Button 
              type="primary" 
              htmlType="submit" 
              loading={uploading}
              icon={<UploadOutlined />}
              size="large"
            >
              {uploading ? '上传中...' : '上传文件'}
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Card title="已上传的数据集">
        <Table
          columns={columns}
          dataSource={datasets}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个数据集`,
          }}
          locale={{ emptyText: '暂无数据集，请上传CSV文件' }}
        />
      </Card>
    </div>
  );
};

export default DataUpload;
