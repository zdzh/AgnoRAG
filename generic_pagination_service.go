package controllers

// Generic version with type safety (requires Go 1.18+)

// GenericResourceService defines methods for resource operations with generic types
type GenericResourceService[T any] interface {
	GetAll(owner string) ([]*T, error)
	GetAllByOrganization(owner, organization string) ([]*T, error)
	GetCount(owner, field, value string) (int64, error)
	GetPaginated(owner string, offset, limit int, field, value, sortField, sortOrder string) ([]*T, error)
	GetMasked(resources []*T, userId string) []*T
}

// GenericPaginationService handles pagination with type safety
type GenericPaginationService[T any] struct{}

// NewGenericPaginationService creates a new instance
func NewGenericPaginationService[T any]() *GenericPaginationService[T] {
	return &GenericPaginationService[T]{}
}

// HandlePaginatedQuery handles pagination with type safety
func (ps *GenericPaginationService[T]) HandlePaginatedQuery(
	c *ApiController,
	config PaginationConfig,
	service GenericResourceService[T],
) {
	userId := c.GetSessionUsername()
	
	if config.Limit == "" || config.Page == "" {
		ps.handleNonPaginatedQuery(c, config, service, userId)
	} else {
		ps.handlePaginatedQuery(c, config, service, userId)
	}
}

func (ps *GenericPaginationService[T]) handleNonPaginatedQuery(
	c *ApiController,
	config PaginationConfig,
	service GenericResourceService[T],
	userId string,
) {
	var resources []*T
	var err error
	
	if config.Organization == "" {
		resources, err = service.GetAll(config.Owner)
	} else {
		resources, err = service.GetAllByOrganization(config.Owner, config.Organization)
	}
	
	if err != nil {
		c.ResponseErr(err)
		return
	}
	
	maskedResources := service.GetMasked(resources, userId)
	c.ResponseOk(maskedResources)
}

func (ps *GenericPaginationService[T]) handlePaginatedQuery(
	c *ApiController,
	config PaginationConfig,
	service GenericResourceService[T],
	userId string,
) {
	limit := util.ParseInt(config.Limit)
	
	count, err := service.GetCount(config.Owner, config.Field, config.Value)
	if err != nil {
		c.ResponseErr(err)
		return
	}
	
	paginator := pagination.SetPaginator(c.Ctx, limit, count)
	
	resources, err := service.GetPaginated(
		config.Owner,
		paginator.Offset(),
		limit,
		config.Field,
		config.Value,
		config.SortField,
		config.SortOrder,
	)
	if err != nil {
		c.ResponseErr(err)
		return
	}
	
	maskedResources := service.GetMasked(resources, userId)
	c.ResponseOk(maskedResources, paginator.Nums())
}

// GenericApplicationService implements GenericResourceService for Applications
type GenericApplicationService struct{}

func (as *GenericApplicationService) GetAll(owner string) ([]*object.Application, error) {
	return object.GetApplications(owner)
}

func (as *GenericApplicationService) GetAllByOrganization(owner, organization string) ([]*object.Application, error) {
	return object.GetOrganizationApplications(owner, organization)
}

func (as *GenericApplicationService) GetCount(owner, field, value string) (int64, error) {
	return object.GetApplicationCount(owner, field, value)
}

func (as *GenericApplicationService) GetPaginated(owner string, offset, limit int, field, value, sortField, sortOrder string) ([]*object.Application, error) {
	return object.GetPaginationApplications(owner, offset, limit, field, value, sortField, sortOrder)
}

func (as *GenericApplicationService) GetMasked(resources []*object.Application, userId string) []*object.Application {
	return object.GetMaskedApplications(resources, userId)
}

// Usage example with generic service
func (c *ApiController) GetApplicationsGeneric() {
	config := c.GetPaginationConfig()
	paginationService := NewGenericPaginationService[object.Application]()
	applicationService := &GenericApplicationService{}
	paginationService.HandlePaginatedQuery(c, config, applicationService)
}

// Factory pattern for creating pagination handlers
type PaginationHandlerFactory struct {
	paginationService *PaginationService
}

func NewPaginationHandlerFactory() *PaginationHandlerFactory {
	return &PaginationHandlerFactory{
		paginationService: NewPaginationService(),
	}
}

func (factory *PaginationHandlerFactory) CreateApplicationHandler() func(*ApiController) {
	return func(c *ApiController) {
		config := c.GetPaginationConfig()
		applicationService := &ApplicationService{}
		factory.paginationService.HandlePaginatedQuery(c, config, applicationService)
	}
}

func (factory *PaginationHandlerFactory) CreateUserHandler() func(*ApiController) {
	return func(c *ApiController) {
		config := c.GetPaginationConfig()
		userService := &UserService{}
		factory.paginationService.HandlePaginatedQuery(c, config, userService)
	}
}

func (factory *PaginationHandlerFactory) CreateRoleHandler() func(*ApiController) {
	return func(c *ApiController) {
		config := c.GetPaginationConfig()
		roleService := &RoleService{}
		factory.paginationService.HandlePaginatedQuery(c, config, roleService)
	}
}

// Usage with factory pattern
func (c *ApiController) GetApplicationsWithFactory() {
	factory := NewPaginationHandlerFactory()
	handler := factory.CreateApplicationHandler()
	handler(c)
}