import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Table, Tag, Progress, Alert } from 'antd';
import { 
  DatabaseOutlined, 
  ExperimentOutlined, 
  RobotOutlined, 
  CodeOutlined,
  TrophyOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';

const Dashboard = () => {
  const [systemStats, setSystemStats] = useState(null);
  const [recentModels, setRecentModels] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // 获取系统统计
      const statsResponse = await axios.get('http://localhost:5001/api/system/stats');
      setSystemStats(statsResponse.data);

      // 获取系统健康状态
      const healthResponse = await axios.get('http://localhost:5001/api/system/health');
      setSystemHealth(healthResponse.data);

      // 获取数据集列表（用于获取最近的模型）
      const datasetsResponse = await axios.get('http://localhost:5001/api/data/datasets');
      if (datasetsResponse.data.datasets.length > 0) {
        const firstDatasetId = datasetsResponse.data.datasets[0].id;
        const modelsResponse = await axios.get(`http://localhost:5001/api/ml/datasets/${firstDatasetId}/models`);
        setRecentModels(modelsResponse.data.models.slice(0, 5));
      }

    } catch (error) {
      console.error('获取仪表板数据失败:', error);
    } finally {
      setLoading(false);
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

  const modelColumns = [
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
        const algorithmNames = {
          'logistic_regression': '逻辑回归',
          'random_forest': '随机森林',
          'svm': '支持向量机',
          'gradient_boosting': '梯度提升'
        };
        return algorithmNames[algorithm] || algorithm;
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
    }
  ];

  if (loading) {
    return (
      <div style={{ padding: 24, textAlign: 'center' }}>
        <Progress type="circle" />
        <p style={{ marginTop: 16 }}>加载仪表板数据...</p>
      </div>
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 24 }}>系统仪表板</h2>
      
      {/* 系统健康状态 */}
      {systemHealth && (
        <Alert
          message={`系统状态: ${systemHealth.status === 'healthy' ? '健康' : '异常'}`}
          description={`最后检查时间: ${new Date(systemHealth.timestamp).toLocaleString()}`}
          type={systemHealth.status === 'healthy' ? 'success' : 'warning'}
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="数据集总数"
              value={systemStats?.database_stats?.total_datasets || 0}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="模型总数"
              value={systemStats?.database_stats?.total_models || 0}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="AI解释次数"
              value={systemStats?.database_stats?.total_explanations || 0}
              prefix={<RobotOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="生成代码数"
              value={systemStats?.database_stats?.total_generated_codes || 0}
              prefix={<CodeOutlined />}
              valueStyle={{ color: '#eb2f96' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统资源使用情况 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={8}>
          <Card title="CPU使用率" size="small">
            <Progress
              type="circle"
              percent={Math.round(systemStats?.system_resources?.cpu_usage_percent || 0)}
              format={percent => `${percent}%`}
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="内存使用率" size="small">
            <Progress
              type="circle"
              percent={Math.round(systemStats?.system_resources?.memory_usage_percent || 0)}
              format={percent => `${percent}%`}
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
            <p style={{ textAlign: 'center', marginTop: 8, fontSize: '12px' }}>
              {systemStats?.system_resources?.memory_used_gb?.toFixed(1)}GB / 
              {systemStats?.system_resources?.memory_total_gb?.toFixed(1)}GB
            </p>
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="磁盘使用率" size="small">
            <Progress
              type="circle"
              percent={Math.round(systemStats?.system_resources?.disk_usage_percent || 0)}
              format={percent => `${percent}%`}
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
            <p style={{ textAlign: 'center', marginTop: 8, fontSize: '12px' }}>
              {systemStats?.system_resources?.disk_used_gb?.toFixed(1)}GB / 
              {systemStats?.system_resources?.disk_total_gb?.toFixed(1)}GB
            </p>
          </Card>
        </Col>
      </Row>

      {/* 模型状态分布 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="模型状态分布" size="small">
            {systemStats?.model_stats?.by_status && Object.keys(systemStats.model_stats.by_status).length > 0 ? (
              Object.entries(systemStats.model_stats.by_status).map(([status, count]) => (
                <div key={status} style={{ marginBottom: 8 }}>
                  <Tag color={getStatusColor(status)}>{getStatusText(status)}</Tag>
                  <span style={{ marginLeft: 8 }}>{count} 个</span>
                </div>
              ))
            ) : (
              <p>暂无模型数据</p>
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="算法使用分布" size="small">
            {systemStats?.model_stats?.by_algorithm && Object.keys(systemStats.model_stats.by_algorithm).length > 0 ? (
              Object.entries(systemStats.model_stats.by_algorithm).map(([algorithm, count]) => {
                const algorithmNames = {
                  'logistic_regression': '逻辑回归',
                  'random_forest': '随机森林',
                  'svm': '支持向量机',
                  'gradient_boosting': '梯度提升'
                };
                return (
                  <div key={algorithm} style={{ marginBottom: 8 }}>
                    <Tag color="blue">{algorithmNames[algorithm] || algorithm}</Tag>
                    <span style={{ marginLeft: 8 }}>{count} 个</span>
                  </div>
                );
              })
            ) : (
              <p>暂无算法数据</p>
            )}
          </Card>
        </Col>
      </Row>

      {/* 最近的模型 */}
      <Card title="最近的模型" size="small">
        <Table
          columns={modelColumns}
          dataSource={recentModels}
          rowKey="id"
          pagination={false}
          size="small"
          locale={{ emptyText: '暂无模型数据' }}
        />
      </Card>
    </div>
  );
};

export default Dashboard;
