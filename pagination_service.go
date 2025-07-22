package controllers

import (
	"fmt"
	"reflect"
)

// PaginationConfig holds the configuration for pagination queries
type PaginationConfig struct {
	Owner      string
	Limit      string
	Page       string
	Field      string
	Value      string
	SortField  string
	SortOrder  string
	Organization string // Optional, for organization-specific queries
}

// ResourceService interface defines methods that resource services must implement
type ResourceService interface {
	// GetAll returns all resources without pagination
	GetAll(owner string) (interface{}, error)
	// GetAllByOrganization returns all resources for a specific organization
	GetAllByOrganization(owner, organization string) (interface{}, error)
	// GetCount returns the total count of resources matching the criteria
	GetCount(owner, field, value string) (int64, error)
	// GetPaginated returns paginated resources
	GetPaginated(owner string, offset, limit int, field, value, sortField, sortOrder string) (interface{}, error)
	// GetMasked applies masking to the resources for the given user
	GetMasked(resources interface{}, userId string) interface{}
}

// PaginationService handles common pagination logic
type PaginationService struct{}

// NewPaginationService creates a new instance of PaginationService
func NewPaginationService() *PaginationService {
	return &PaginationService{}
}

// HandlePaginatedQuery handles the pagination logic for any resource type
func (ps *PaginationService) HandlePaginatedQuery(
	c *ApiController,
	config PaginationConfig,
	service ResourceService,
) {
	userId := c.GetSessionUsername()
	
	// Check if pagination parameters are provided
	if config.Limit == "" || config.Page == "" {
		// Return all resources without pagination
		ps.handleNonPaginatedQuery(c, config, service, userId)
	} else {
		// Return paginated resources
		ps.handlePaginatedQuery(c, config, service, userId)
	}
}

// handleNonPaginatedQuery handles queries without pagination
func (ps *PaginationService) handleNonPaginatedQuery(
	c *ApiController,
	config PaginationConfig,
	service ResourceService,
	userId string,
) {
	var resources interface{}
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

// handlePaginatedQuery handles queries with pagination
func (ps *PaginationService) handlePaginatedQuery(
	c *ApiController,
	config PaginationConfig,
	service ResourceService,
	userId string,
) {
	limit := util.ParseInt(config.Limit)
	
	// Get total count
	count, err := service.GetCount(config.Owner, config.Field, config.Value)
	if err != nil {
		c.ResponseErr(err)
		return
	}
	
	// Set up paginator
	paginator := pagination.SetPaginator(c.Ctx, limit, count)
	
	// Get paginated resources
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
	
	// Apply masking and return response
	maskedResources := service.GetMasked(resources, userId)
	c.ResponseOk(maskedResources, paginator.Nums())
}

// ApplicationService implements ResourceService for Application resources
type ApplicationService struct{}

func (as *ApplicationService) GetAll(owner string) (interface{}, error) {
	return object.GetApplications(owner)
}

func (as *ApplicationService) GetAllByOrganization(owner, organization string) (interface{}, error) {
	return object.GetOrganizationApplications(owner, organization)
}

func (as *ApplicationService) GetCount(owner, field, value string) (int64, error) {
	return object.GetApplicationCount(owner, field, value)
}

func (as *ApplicationService) GetPaginated(owner string, offset, limit int, field, value, sortField, sortOrder string) (interface{}, error) {
	return object.GetPaginationApplications(owner, offset, limit, field, value, sortField, sortOrder)
}

func (as *ApplicationService) GetMasked(resources interface{}, userId string) interface{} {
	if applications, ok := resources.([]*object.Application); ok {
		return object.GetMaskedApplications(applications, userId)
	}
	return resources
}