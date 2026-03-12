# Архитектура системы шаблонов для GAN конфигурации и фильтров генерации

## Обзор

Система шаблонов позволяет сохранять, управлять и повторно использовать конфигурации GAN и фильтры генерации синтетических данных.

## База данных

### Новые таблицы

#### 1. `gan_config_templates`
```sql
CREATE TABLE gan_config_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    config_json JSONB NOT NULL,  -- Содержит параметры: LATENT_DIM, BATCH_SIZE, LEARNING_RATE, etc.
    created_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Пример config_json:**
```json
{
  "LATENT_DIM": 256,
  "BATCH_SIZE": 1024,
  "LEARNING_RATE": 0.0001,
  "DROPOUT_RATE": 0.1,
  "LAMBDA_GP": 10,
  "N_CRITIC": 5,
  "GENERATOR_LAYERS": [512, 512, 256, 256, 128],
  "DISCRIMINATOR_LAYERS": [512, 512, 256, 256, 128],
  "USE_WGAN_GP": true
}
```

#### 2. `filter_templates`
```sql
CREATE TABLE filter_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    filters_json JSONB NOT NULL,  -- Содержит фильтры: cities, devices, os, numeric_ranges, etc.
    created_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Пример filters_json:**
```json
{
  "cities": ["Москва", "Санкт-Петербург"],
  "devices": ["iPhone"],
  "os": ["iOS 17"],
  "user_types": ["Premium"],
  "numeric_ranges": {
    "age": {"min": 25, "max": 45},
    "session_count": {"min": 10}
  }
}
```

## Backend API

### CRUD операции для шаблонов GAN конфигурации

```python
# backend/database/crud.py

def create_gan_config_template(
    db: Session,
    *,
    name: str,
    description: Optional[str],
    config_json: dict,
    created_by_user_id: Optional[int],
    is_default: bool = False
) -> GANConfigTemplateORM:
    """Создать шаблон конфигурации GAN"""
    pass

def list_gan_config_templates(db: Session, limit: int = 100) -> list[GANConfigTemplateORM]:
    """Получить список шаблонов конфигурации GAN"""
    pass

def get_gan_config_template_by_name(db: Session, name: str) -> Optional[GANConfigTemplateORM]:
    """Получить шаблон по имени"""
    pass

def update_gan_config_template(db: Session, template_id: int, **kwargs) -> Optional[GANConfigTemplateORM]:
    """Обновить шаблон"""
    pass

def delete_gan_config_template(db: Session, template_id: int) -> bool:
    """Удалить шаблон"""
    pass
```

### API эндпоинты

```python
# backend/api/routes/templates.py

@router.get("/gan-config-templates", summary="Список шаблонов конфигурации GAN")
async def list_gan_config_templates(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("developer", "analyst"))
):
    """Возвращает список всех шаблонов конфигурации GAN"""
    pass

@router.post("/gan-config-templates", summary="Создать шаблон конфигурации GAN")
async def create_gan_config_template(
    request: GANConfigTemplateCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("developer", "analyst"))
):
    """Создает новый шаблон конфигурации GAN"""
    pass

@router.put("/gan-config-templates/{template_id}", summary="Обновить шаблон")
async def update_gan_config_template(
    template_id: int,
    request: GANConfigTemplateUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("developer", "analyst"))
):
    """Обновляет существующий шаблон"""
    pass

@router.delete("/gan-config-templates/{template_id}", summary="Удалить шаблон")
async def delete_gan_config_template(
    template_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("developer", "analyst"))
):
    """Удаляет шаблон конфигурации GAN"""
    pass

# Аналогичные эндпоинты для filter-templates
```

## Frontend интеграция

### API клиент

```typescript
// frontend/src/utils/api.ts

export const templatesAPI = {
  // GAN Config Templates
  listGANConfigTemplates: () => api.get('/templates/gan-config-templates'),
  createGANConfigTemplate: (data: any) => api.post('/templates/gan-config-templates', data),
  updateGANConfigTemplate: (id: number, data: any) => api.put(`/templates/gan-config-templates/${id}`, data),
  deleteGANConfigTemplate: (id: number) => api.delete(`/templates/gan-config-templates/${id}`),
  
  // Filter Templates
  listFilterTemplates: () => api.get('/templates/filter-templates'),
  createFilterTemplate: (data: any) => api.post('/templates/filter-templates', data),
  updateFilterTemplate: (id: number, data: any) => api.put(`/templates/filter-templates/${id}`, data),
  deleteFilterTemplate: (id: number) => api.delete(`/templates/filter-templates/${id}`),
};
```

### Интеграция в GANManager

#### 1. Добавление выбора шаблона в форму конфигурации GAN

```tsx
// Над секцией "Переопределение конфигурации"

<Row gutter={16} style={{ marginBottom: 16 }}>
  <Col span={24}>
    <Space>
      <Select
        style={{ width: 300 }}
        placeholder="Загрузить из шаблона"
        allowClear
        options={ganConfigTemplates.map(t => ({ label: t.name, value: t.id }))}
        onChange={(templateId) => {
          const template = ganConfigTemplates.find(t => t.id === templateId);
          if (template) {
            // Заполнить форму значениями из шаблона
            configForm.setFieldsValue({
              LATENT_DIM: template.config_json.LATENT_DIM,
              BATCH_SIZE: template.config_json.BATCH_SIZE,
              LEARNING_RATE: template.config_json.LEARNING_RATE,
              // ... остальные поля
            });
            message.success(`Конфигурация "${template.name}" загружена`);
          }
        }}
      />
      <Button 
        icon={<SaveOutlined />}
        onClick={() => {
          // Открыть модалку для сохранения текущей конфигурации как шаблона
          setShowSaveTemplateModal(true);
        }}
      >
        Сохранить как шаблон
      </Button>
    </Space>
  </Col>
</Row>

<Divider orientation="left">Переопределение конфигурации</Divider>
```

#### 2. Добавление выбора шаблона в фильтры генерации

```tsx
// Над секцией "Фильтры генерации"

<Row gutter={16} style={{ marginBottom: 16 }}>
  <Col span={24}>
    <Space>
      <Select
        style={{ width: 300 }}
        placeholder="Загрузить фильтры из шаблона"
        allowClear
        options={filterTemplates.map(t => ({ label: t.name, value: t.id }))}
        onChange={(templateId) => {
          const template = filterTemplates.find(t => t.id === templateId);
          if (template) {
            // Заполнить filterDraft значениями из шаблона
            setFilterDraft(template.filters_json);
            message.success(`Фильтры "${template.name}" загружены`);
          }
        }}
      />
      <Button 
        icon={<SaveOutlined />}
        onClick={() => {
          // Открыть модалку для сохранения текущих фильтров как шаблона
          setShowSaveFilterTemplateModal(true);
        }}
      >
        Сохранить фильтры как шаблон
      </Button>
    </Space>
  </Col>
</Row>

<Divider orientation="left">Фильтры генерации</Divider>
```

### Страница Шаблоны

```tsx
// frontend/src/components/Templates/TemplatesPage.tsx

export const TemplatesPage: React.FC = () => {
  const [ganConfigTemplates, setGanConfigTemplates] = useState<any[]>([]);
  const [filterTemplates, setFilterTemplates] = useState<any[]>([]);
  
  return (
    <div style={{ padding: '20px' }}>
      <Tabs
        items={[
          {
            key: 'gan-config',
            label: 'Шаблоны конфигурации GAN',
            children: (
              <>
                <Button 
                  type="primary" 
                  icon={<PlusOutlined />}
                  onClick={() => setShowCreateGANTemplateModal(true)}
                  style={{ marginBottom: 16 }}
                >
                  Создать шаблон
                </Button>
                <Table
                  dataSource={ganConfigTemplates}
                  columns={[
                    { title: 'Название', dataIndex: 'name' },
                    { title: 'Описание', dataIndex: 'description' },
                    { title: 'Создан', dataIndex: 'created_at', render: (v) => new Date(v).toLocaleString() },
                    {
                      title: 'Действия',
                      render: (_, record) => (
                        <Space>
                          <Button type="link" onClick={() => handleViewTemplate(record)}>Просмотр</Button>
                          <Button type="link" onClick={() => handleEditTemplate(record)}>Редактировать</Button>
                          <Popconfirm title="Удалить?" onConfirm={() => handleDeleteTemplate(record.id)}>
                            <Button type="link" danger>Удалить</Button>
                          </Popconfirm>
                        </Space>
                      )
                    }
                  ]}
                />
              </>
            ),
          },
          {
            key: 'filters',
            label: 'Шаблоны фильтров',
            children: (
              // Аналогично для фильтров
            ),
          },
        ]}
      />
    </div>
  );
};
```

## Примеры использования

### Workflow 1: Использование готового шаблона GAN конфигурации

1. Пользователь открывает "Конфигурация и обучение GAN"
2. Выбирает из выпадающего списка шаблон "Optimal WGAN-GP для больших данных"
3. Форма автоматически заполняется значениями из шаблона
4. Пользователь может дополнительно изменить параметры
5. Запускает обучение

### Workflow 2: Сохранение текущей конфигурации как шаблона

1. Пользователь настраивает параметры GAN вручную
2. Нажимает "Сохранить как шаблон"
3. В модалке вводит:
   - Название: "iPhone users Moscow config"
   - Описание: "Оптимальная конфигурация для генерации пользователей iPhone в Москве"
   - Отметить как дефолтный (опционально)
4. Сохраняет шаблон
5. Теперь этот шаблон доступен в списке

### Workflow 3: Использование шаблонов фильтров

1. Пользователь открывает "Генерация синтетических данных"
2. Выбирает из списка шаблон "Premium iOS users Moscow-SPb"
3. Фильтры автоматически применяются
4. Генерирует данные

## Преимущества архитектуры

1. **Переиспользование**: Один раз настроил - используй многократно
2. **Консистентность**: Команда использует одинаковые конфигурации
3. **Документирование**: Описание помогает понять, для чего нужен шаблон
4. **Быстрота**: Не нужно вручную вводить десятки параметров
5. **Версионирование**: История изменений через updated_at
6. **Управление правами**: Через created_by_user_id можно отслеживать автора

## Расширения (опционально)

1. **Теги для шаблонов**: Добавить поле `tags: string[]` для группировки
2. **Версионирование шаблонов**: Хранить историю изменений
3. **Шаринг шаблонов**: Публичные/приватные шаблоны
4. **Валидация**: Проверка корректности параметров при создании шаблона
5. **Импорт/Экспорт**: JSON файлы для переноса между окружениями
