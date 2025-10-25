{{/*
Expand the name of the chart.
*/}}
{{- define "flask-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "flask-app.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "flask-app.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "flask-app.labels" -}}
helm.sh/chart: {{ include "flask-app.chart" . }}
{{ include "flask-app.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- if .Values.flask.env }}
app.kubernetes.io/environment: {{ .Values.flask.env }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "flask-app.selectorLabels" -}}
app.kubernetes.io/name: {{ include "flask-app.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "flask-app.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "flask-app.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate database connection URL
*/}}
{{- define "flask-app.databaseURL" -}}
{{- if .Values.database.enabled }}
{{- printf "postgresql://%s:%s@%s:%s/%s?sslmode=%s" .Values.database.username .Values.database.password .Values.database.host (.Values.database.port | toString) .Values.database.name .Values.database.sslMode }}
{{- end }}
{{- end }}

{{/*
Generate Redis URL
*/}}
{{- define "flask-app.redisURL" -}}
{{- if .Values.redis.enabled }}
{{- if .Values.redis.password }}
{{- printf "redis://:%s@%s:%s/%s" .Values.redis.password .Values.redis.host (.Values.redis.port | toString) (.Values.redis.database | toString) }}
{{- else }}
{{- printf "redis://%s:%s/%s" .Values.redis.host (.Values.redis.port | toString) (.Values.redis.database | toString) }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the proper Docker Image Registry Secret Names
*/}}
{{- define "flask-app.imagePullSecrets" -}}
{{- if .Values.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return true if a secret object should be created for database credentials
*/}}
{{- define "flask-app.createDatabaseSecret" -}}
{{- if and .Values.database.enabled (not .Values.database.existingSecret) }}
    {{- true -}}
{{- end }}
{{- end }}

{{/*
Get the database secret name
*/}}
{{- define "flask-app.databaseSecretName" -}}
{{- if .Values.database.existingSecret }}
    {{- .Values.database.existingSecret }}
{{- else }}
    {{- printf "%s-db" (include "flask-app.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Get the Flask secret name
*/}}
{{- define "flask-app.flaskSecretName" -}}
{{- if .Values.flask.existingSecret }}
    {{- .Values.flask.existingSecret }}
{{- else }}
    {{- printf "%s-flask" (include "flask-app.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Renders a value that contains template.
Usage:
{{ include "flask-app.tplValue" ( dict "value" .Values.path.to.the.Value "context" $) }}
*/}}
{{- define "flask-app.tplValue" -}}
    {{- if typeIs "string" .value }}
        {{- tpl .value .context }}
    {{- else }}
        {{- tpl (.value | toYaml) .context }}
    {{- end }}
{{- end -}}

{{/*
Return the appropriate apiVersion for HPA
*/}}
{{- define "flask-app.hpa.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "autoscaling/v2" }}
{{- print "autoscaling/v2" }}
{{- else }}
{{- print "autoscaling/v2beta2" }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for PodDisruptionBudget
*/}}
{{- define "flask-app.pdb.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "policy/v1/PodDisruptionBudget" }}
{{- print "policy/v1" }}
{{- else }}
{{- print "policy/v1beta1" }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for Ingress
*/}}
{{- define "flask-app.ingress.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "networking.k8s.io/v1/Ingress" }}
{{- print "networking.k8s.io/v1" }}
{{- else if .Capabilities.APIVersions.Has "networking.k8s.io/v1beta1/Ingress" }}
{{- print "networking.k8s.io/v1beta1" }}
{{- else }}
{{- print "extensions/v1beta1" }}
{{- end }}
{{- end }}

{{/*
Compile all warnings into a single message
*/}}
{{- define "flask-app.validateValues" -}}
{{- $messages := list -}}
{{- if and .Values.autoscaling.enabled (not (or .Values.autoscaling.targetCPUUtilizationPercentage .Values.autoscaling.targetMemoryUtilizationPercentage)) -}}
{{- $messages = append $messages "ERROR: When autoscaling is enabled, at least one of targetCPUUtilizationPercentage or targetMemoryUtilizationPercentage must be set" -}}
{{- end -}}
{{- if and .Values.database.enabled (not .Values.database.password) (not .Values.postgresql.enabled) -}}
{{- $messages = append $messages "WARNING: database.enabled is true but database.password is empty. A random password will be generated." -}}
{{- end -}}
{{- if and .Values.persistence.enabled (not .Values.persistence.storageClass) -}}
{{- $messages = append $messages "INFO: persistence.enabled is true but storageClass is empty. Using cluster default." -}}
{{- end -}}
{{- if and (eq .Values.flask.env "production") .Values.flask.debug -}}
{{- $messages = append $messages "WARNING: Debug mode is enabled in production environment. This is not recommended!" -}}
{{- end -}}
{{- if and (eq .Values.flask.env "production") (eq .Values.flask.secretKey "change-me-in-production-use-vault-or-sealed-secrets") -}}
{{- $messages = append $messages "ERROR: Default Flask secret key is being used in production. Please change it!" -}}
{{- end -}}
{{- range $messages }}
{{ . }}
{{- end -}}
{{- end -}}
