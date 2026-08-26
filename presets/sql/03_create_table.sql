-- 用户相关
CREATE TABLE IF NOT EXISTS users
(
    id              SERIAL,
    username        VARCHAR NOT NULL,
    email           VARCHAR NOT NULL,
    role            INTEGER NOT NULL,
    hashed_password VARCHAR NOT NULL,
    is_active       BOOLEAN NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (email)
);

-- 设置相关
CREATE TYPE providertype AS ENUM ('OPENAI', 'ANTHROPIC', 'GOOGLE');
CREATE TABLE IF NOT EXISTS llm_providers
(
    id            SERIAL,
    name          VARCHAR      NOT NULL,
    provider_type providertype NOT NULL,
    base_url      VARCHAR,
    api_key       VARCHAR,
    use_proxy     BOOLEAN      NOT NULL,
    is_active     BOOLEAN      NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS llm_models
(
    id               SERIAL,
    provider_id      INTEGER,
    display_name     VARCHAR NOT NULL,
    name             VARCHAR NOT NULL,
    input_price      NUMERIC NOT NULL,
    output_price     NUMERIC NOT NULL,
    cache_read_price NUMERIC NOT NULL,
    reasoning_effort VARCHAR,
    is_active        BOOLEAN NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (provider_id) REFERENCES llm_providers
);

CREATE TABLE IF NOT EXISTS llm_settings
(
    id                        INTEGER NOT NULL DEFAULT 1,
    chat_model_id             INTEGER,
    vision_model_id           INTEGER,
    agent_model_id            INTEGER,
    coding_model_id           INTEGER,
    simple_model_id           INTEGER,
    long_context_model_id     INTEGER,
    high_performance_model_id INTEGER,
    PRIMARY KEY (id),
    FOREIGN KEY (chat_model_id) REFERENCES llm_models,
    FOREIGN KEY (vision_model_id) REFERENCES llm_models,
    FOREIGN KEY (agent_model_id) REFERENCES llm_models,
    FOREIGN KEY (coding_model_id) REFERENCES llm_models,
    FOREIGN KEY (simple_model_id) REFERENCES llm_models,
    FOREIGN KEY (long_context_model_id) REFERENCES llm_models,
    FOREIGN KEY (high_performance_model_id) REFERENCES llm_models
);

CREATE TABLE IF NOT EXISTS llm_statistics
(
    id            SERIAL,
    agent_name    VARCHAR NOT NULL,
    model_id      INTEGER,
    setting_key   VARCHAR,
    user_id       INTEGER,
    thread        VARCHAR,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read    INTEGER,
    price         NUMERIC NOT NULL,
    time          TIMESTAMP,
    PRIMARY KEY (id),
    FOREIGN KEY (model_id) REFERENCES llm_models,
    FOREIGN KEY (user_id) REFERENCES users
);


-- 项目相关
CREATE TABLE IF NOT EXISTS projects
(
    id          SERIAL,
    owner_id    INTEGER                  NOT NULL,
    name        VARCHAR                  NOT NULL,
    description VARCHAR                  NOT NULL,
    public      BOOLEAN                  NOT NULL,
    create_time TIMESTAMP WITH TIME ZONE NOT NULL,
    update_time TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (owner_id) REFERENCES users
);

-- 可持久执行和重放的 Web 智能体会话
CREATE TABLE IF NOT EXISTS agent_sessions
(
    uid         uuid                     NOT NULL,
    owner_id    INTEGER                  NOT NULL,
    project_id  INTEGER,
    title       VARCHAR(200)             NOT NULL DEFAULT '',
    create_time TIMESTAMP WITH TIME ZONE NOT NULL,
    update_time TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (owner_id) REFERENCES users ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_runs
(
    uid                  uuid                     NOT NULL,
    session_uid          uuid                     NOT NULL,
    project_id           INTEGER,
    agent_name           VARCHAR(64)              NOT NULL,
    status               VARCHAR(32)              NOT NULL DEFAULT 'queued',
    checkpoint_thread_id VARCHAR(64)              NOT NULL,
    celery_task_id       VARCHAR(64),
    interrupt_payload    jsonb,
    result_payload       jsonb,
    error_code           VARCHAR(64),
    error_message        TEXT,
    cancel_requested     BOOLEAN                  NOT NULL DEFAULT FALSE,
    retry_of_uid         uuid,
    heartbeat_at         TIMESTAMP WITH TIME ZONE,
    create_time          TIMESTAMP WITH TIME ZONE NOT NULL,
    start_time           TIMESTAMP WITH TIME ZONE,
    end_time             TIMESTAMP WITH TIME ZONE,
    update_time          TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (uid),
    UNIQUE (checkpoint_thread_id),
    FOREIGN KEY (session_uid) REFERENCES agent_sessions ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects ON DELETE SET NULL,
    FOREIGN KEY (retry_of_uid) REFERENCES agent_runs ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS agent_messages
(
    uid         uuid                     NOT NULL,
    session_uid uuid                     NOT NULL,
    run_uid     uuid                     NOT NULL,
    role        VARCHAR(16)              NOT NULL,
    parts       jsonb                    NOT NULL DEFAULT '[]'::jsonb,
    create_time TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (session_uid) REFERENCES agent_sessions ON DELETE CASCADE,
    FOREIGN KEY (run_uid) REFERENCES agent_runs ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_events
(
    id          BIGSERIAL,
    run_uid     uuid                     NOT NULL,
    event_type  VARCHAR(64)              NOT NULL,
    payload     jsonb                    NOT NULL DEFAULT '{}'::jsonb,
    create_time TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (run_uid) REFERENCES agent_runs ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS project_tags
(
    id    SERIAL,
    name  VARCHAR NOT NULL,
    color VARCHAR NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS project_tag_link
(
    project_id INTEGER NOT NULL,
    tag_id     INTEGER NOT NULL,
    PRIMARY KEY (project_id, tag_id),
    FOREIGN KEY (project_id) REFERENCES projects ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES project_tags
);

CREATE TABLE IF NOT EXISTS project_star
(
    project_id INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    PRIMARY KEY (project_id),
    FOREIGN KEY (project_id) REFERENCES projects ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users
);

-- 资源相关
CREATE TABLE IF NOT EXISTS bio_dbs
(
    id          SERIAL,
    name        VARCHAR NOT NULL,
    description VARCHAR,
    path        VARCHAR,
    size        VARCHAR,
    last_update VARCHAR,
    download_command VARCHAR,
    download_image VARCHAR,
    PRIMARY KEY (id),
    UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS reference_genomes
(
    id               SERIAL,
    ncbi_tax_id      INTEGER NOT NULL,
    name             VARCHAR NOT NULL,
    accession        VARCHAR NOT NULL,
    aliases          VARCHAR,
    genome_sequences VARCHAR NOT NULL,
    annotation_gff   VARCHAR NOT NULL,
    annotation_gtf   VARCHAR NOT NULL,
    bowtie2_index    VARCHAR,
    bwa_index        VARCHAR,
    hisat2_index     VARCHAR,
    star_index       VARCHAR,
    minimap2_index   VARCHAR,
    PRIMARY KEY (id),
    UNIQUE (ncbi_tax_id),
    UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS samples
(
    uid         uuid                     NOT NULL,
    owner_id    INTEGER                  NOT NULL,
    project_id  INTEGER                  NOT NULL,
    sample_name VARCHAR                  NOT NULL,
    meta_data   jsonb,
    create_time TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (owner_id) REFERENCES users,
    FOREIGN KEY (project_id) REFERENCES projects ON DELETE CASCADE,
    CONSTRAINT uq_sample_project_name UNIQUE (project_id, sample_name)
);

CREATE TABLE IF NOT EXISTS sample_files
(
    uid                   uuid                     NOT NULL,
    sample_uid            uuid,
    tag                   VARCHAR(50),
    file_path             VARCHAR                  NOT NULL,
    file_format           VARCHAR,
    file_size             BIGINT                   NOT NULL,
    md5_checksum          VARCHAR,
    description           VARCHAR                  NOT NULL DEFAULT '',
    is_dynamic            BOOLEAN                  NOT NULL DEFAULT FALSE,
    producer_workflow_uid uuid,
    producer_run_uid      uuid,
    uploaded_time         TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (sample_uid) REFERENCES samples ON DELETE CASCADE,
    CONSTRAINT uq_sample_file_tag UNIQUE (sample_uid, tag)
);

CREATE TABLE IF NOT EXISTS project_file_mappings
(
    id                    SERIAL,
    project_id            INTEGER                  NOT NULL,
    keyword               VARCHAR(50)              NOT NULL,
    file_path             VARCHAR(500)             NOT NULL,
    description           VARCHAR                  NOT NULL,
    is_dynamic            BOOLEAN                  NOT NULL DEFAULT FALSE,
    producer_workflow_uid uuid,
    producer_run_uid      uuid,
    create_time           TIMESTAMP WITH TIME ZONE NOT NULL,
    update_time           TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (project_id) REFERENCES projects ON DELETE CASCADE,
    CONSTRAINT uq_project_file_mapping UNIQUE (project_id, keyword)
);

-- 工具相关
CREATE TABLE IF NOT EXISTS images
(
    uid         uuid    NOT NULL,
    name        VARCHAR NOT NULL,
    aliases     VARCHAR[] NOT NULL DEFAULT ARRAY[]::VARCHAR[],
    version     VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    homepage    VARCHAR,
    paper_link  VARCHAR,
    image       jsonb,
    PRIMARY KEY (uid)
);

CREATE TABLE IF NOT EXISTS tool_docs
(
    uid          uuid    NOT NULL,
    help_command VARCHAR NOT NULL,
    description  VARCHAR,
    content      VARCHAR NOT NULL,
    image_uid    uuid,
    PRIMARY KEY (uid),
    FOREIGN KEY (image_uid) REFERENCES images
);

CREATE TABLE IF NOT EXISTS tool_tags
(
    id   SERIAL,
    name VARCHAR NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS tool_groups
(
    id        SERIAL,
    name      VARCHAR NOT NULL,
    parent_id INTEGER,
    tag_id    INTEGER,
    PRIMARY KEY (id),
    FOREIGN KEY (parent_id) REFERENCES tool_groups,
    FOREIGN KEY (tag_id) REFERENCES tool_tags
);

CREATE TABLE IF NOT EXISTS tools
(
    uid                      uuid    NOT NULL,
    name                     VARCHAR NOT NULL,
    image_uid                uuid,
    description              VARCHAR NOT NULL,
    tool_type                INTEGER NOT NULL,
    group_id                 INTEGER,
    help_doc_uid             uuid,
    command_template         VARCHAR NOT NULL,
    dynamic_params           jsonb,
    immutable_static_params  VARCHAR,
    modifiable_static_params VARCHAR,
    file_mounts              jsonb,
    PRIMARY KEY (uid),
    FOREIGN KEY (image_uid) REFERENCES images,
    FOREIGN KEY (group_id) REFERENCES tool_groups,
    FOREIGN KEY (help_doc_uid) REFERENCES tool_docs
);

CREATE TABLE IF NOT EXISTS tool_tag_link
(
    tool_uid uuid    NOT NULL,
    tag_id   INTEGER NOT NULL,
    PRIMARY KEY (tool_uid, tag_id),
    FOREIGN KEY (tool_uid) REFERENCES tools,
    FOREIGN KEY (tag_id) REFERENCES tool_tags
);

-- 工作流相关
CREATE TABLE IF NOT EXISTS workflows
(
    uid             uuid    NOT NULL,
    name            VARCHAR NOT NULL,
    owner_id        INTEGER NOT NULL,
    description     VARCHAR NOT NULL,
    wf_type         INTEGER NOT NULL,
    execution_scope INTEGER NOT NULL,
    workflow        jsonb,
    public          BOOLEAN NOT NULL,
    auto_summary    BOOLEAN NOT NULL DEFAULT FALSE,
    summary_prompt  VARCHAR,
    PRIMARY KEY (uid),
    FOREIGN KEY (owner_id) REFERENCES users
);

CREATE TABLE IF NOT EXISTS project_workflows
(
    project_id   INTEGER                  NOT NULL,
    workflow_uid uuid                     NOT NULL,
    import_time  TIMESTAMP WITH TIME ZONE NOT NULL,
    enabled      BOOLEAN                  NOT NULL,
    PRIMARY KEY (project_id, workflow_uid),
    FOREIGN KEY (project_id) REFERENCES projects ON DELETE CASCADE,
    FOREIGN KEY (workflow_uid) REFERENCES workflows (uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_instances
(
    uid          uuid    NOT NULL,
    name         VARCHAR NOT NULL,
    owner_id     INTEGER NOT NULL,
    project_id   INTEGER,
    workflow_uid uuid,
    sample_uid   uuid,
    base_dir     VARCHAR,
    workflow     jsonb,
    status       INTEGER NOT NULL,
    create_time  TIMESTAMP WITH TIME ZONE,
    start_time   TIMESTAMP WITH TIME ZONE,
    end_time     TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (uid),
    FOREIGN KEY (owner_id) REFERENCES users,
    FOREIGN KEY (project_id) REFERENCES projects ON DELETE CASCADE,
    FOREIGN KEY (sample_uid) REFERENCES samples ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS task_instances
(
    uid          uuid    NOT NULL,
    name         VARCHAR NOT NULL,
    instance_uid uuid    NOT NULL,
    owner_id     INTEGER NOT NULL,
    tool_uid     uuid,
    commands     VARCHAR,
    base_dir     VARCHAR,
    tool_output  jsonb,
    system       VARCHAR,
    hostname     VARCHAR,
    status       INTEGER NOT NULL,
    create_time  TIMESTAMP WITH TIME ZONE,
    start_time   TIMESTAMP WITH TIME ZONE,
    end_time     TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (uid),
    FOREIGN KEY (instance_uid) REFERENCES run_instances ON DELETE CASCADE,
    FOREIGN KEY (owner_id) REFERENCES users,
    FOREIGN KEY (tool_uid) REFERENCES tools
);

CREATE TABLE IF NOT EXISTS task_monitor_logs
(
    id        SERIAL,
    task_uid  uuid,
    cpu_usage DOUBLE PRECISION NOT NULL,
    mem_usage DOUBLE PRECISION NOT NULL,
    mem_used  DOUBLE PRECISION NOT NULL,
    io_in     DOUBLE PRECISION NOT NULL,
    io_out    DOUBLE PRECISION NOT NULL,
    time      TIMESTAMP        NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (task_uid) REFERENCES task_instances ON DELETE CASCADE
);

-- 文档相关
CREATE TABLE IF NOT EXISTS parent_documents
(
    uid         uuid                     NOT NULL,
    doc_id      VARCHAR                  NOT NULL,
    content     TEXT                     NOT NULL,
    title       VARCHAR,
    section     VARCHAR,
    create_time TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (uid)
);

CREATE TABLE IF NOT EXISTS files
(
    id          SERIAL                   NOT NULL,
    sha256      VARCHAR                  NOT NULL,
    file_name   VARCHAR                  NOT NULL,
    s3_path     VARCHAR                  NOT NULL,
    upload_time TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (sha256)
);

-- 系统通知渠道
CREATE TABLE IF NOT EXISTS notification_channels
(
    id                SERIAL                   NOT NULL,
    name              VARCHAR(120)             NOT NULL,
    provider          VARCHAR(32)              NOT NULL,
    enabled           BOOLEAN                  NOT NULL DEFAULT TRUE,
    config            jsonb                    NOT NULL DEFAULT '{}'::jsonb,
    secret_ciphertext TEXT                     NOT NULL,
    last_attempt_at   TIMESTAMP WITH TIME ZONE,
    last_success_at   TIMESTAMP WITH TIME ZONE,
    last_error        TEXT,
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS notification_subscriptions
(
    id         SERIAL       NOT NULL,
    channel_id INTEGER      NOT NULL,
    event_type VARCHAR(120) NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (channel_id, event_type),
    FOREIGN KEY (channel_id) REFERENCES notification_channels ON DELETE CASCADE
);
